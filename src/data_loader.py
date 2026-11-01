"""Data loader, validation models, and persistence utilities for SkillShare Match."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator, model_validator

# Allowed literals
SessionMode = Literal["online", "in_person", "both"]
ConstraintRule = Literal["required", "preferred"]
ConstraintField = Literal["language", "availability", "session_mode", "location_area"]
QueryCategory = Literal["paraphrased", "vague", "multi_skill", "code_mixed", "no_match"]
QuerySplit = Literal["validation", "test"]
RelevanceGrade = Literal[0, 1, 2]


class Constraint(BaseModel):
    """Specification of a practical constraint on the search."""
    field: ConstraintField
    value: Union[str, List[str]]
    rule: ConstraintRule = "preferred"

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, str):
            clean_str = v.strip()
            if not clean_str:
                raise ValueError("Constraint value string cannot be empty.")
            return clean_str
        elif isinstance(v, list):
            clean_list = [item.strip() for item in v if isinstance(item, str) and item.strip()]
            if not clean_list:
                raise ValueError("Constraint value list cannot be empty.")
            return clean_list
        raise ValueError("Constraint value must be a string or list of strings.")


class Profile(BaseModel):
    """Fictional community helper profile."""
    helper_id: str = Field(..., description="Unique stable helper identifier, e.g. H001")
    name: str = Field(..., description="Fictional display name, e.g. Harpreet K.")
    skills: List[str] = Field(..., description="List of structured skill tags")
    bio: str = Field(..., description="Natural language bio description")
    languages: List[str] = Field(..., description="Languages spoken by helper")
    availability: List[str] = Field(..., description="Days/times helper is available")
    session_mode: SessionMode = Field(..., description="online, in_person, or both")
    location_area: Optional[str] = Field(default=None, description="Broad geographic area")
    active: bool = Field(default=True, description="Whether profile is eligible for recommendations")

    @field_validator("helper_id")
    @classmethod
    def validate_helper_id(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("helper_id cannot be empty.")
        return clean

    @field_validator("name", "bio")
    @classmethod
    def validate_non_empty_str(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Text field cannot be empty.")
        return clean

    @field_validator("skills", "languages", "availability", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            cleaned = [str(item).strip() for item in v if str(item).strip()]
            if not cleaned:
                raise ValueError("List field must contain at least one non-empty item.")
            return cleaned
        if isinstance(v, str):
            # Support pipe, comma, or json string
            raw = v.strip()
            if not raw:
                raise ValueError("List field cannot be empty string.")
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    parsed = json.loads(raw)
                    return [str(x).strip() for x in parsed if str(x).strip()]
                except json.JSONDecodeError:
                    pass
            delimiter = "|" if "|" in raw else ","
            parts = [p.strip() for p in raw.split(delimiter) if p.strip()]
            if not parts:
                raise ValueError("List field must contain at least one element.")
            return parts
        raise ValueError(f"Unsupported format for list field: {type(v)}")

    @field_validator("active", mode="before")
    @classmethod
    def parse_active_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in {"true", "1", "yes", "y"}:
                return True
            if clean in {"false", "0", "no", "n"}:
                return False
        if isinstance(v, (int, float)):
            return bool(v)
        raise ValueError(f"Cannot parse {v} as boolean.")


class Query(BaseModel):
    """Evaluation request query."""
    query_id: str = Field(..., description="Unique query identifier, e.g. Q001")
    text: str = Field(..., description="Free-text need description")
    category: QueryCategory = Field(..., description="paraphrased, vague, multi_skill, code_mixed, or no_match")
    constraints: List[Constraint] = Field(default_factory=list, description="Extracted or stated constraints")
    relevant_helper_ids: List[str] = Field(default_factory=list, description="Ground truth relevant helper IDs")
    split: QuerySplit = Field(..., description="validation or test")

    @field_validator("query_id", "text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Field cannot be empty.")
        return clean

    @model_validator(mode="after")
    def validate_no_match_empty_relevance(self) -> "Query":
        if self.category == "no_match" and len(self.relevant_helper_ids) > 0:
            raise ValueError(
                f"Query {self.query_id} is categorized as 'no_match' but has non-empty relevant_helper_ids: "
                f"{self.relevant_helper_ids}"
            )
        return self


class RelevanceLabel(BaseModel):
    """Fine-grained graded relevance annotation for a query-helper pair."""
    query_id: str
    helper_id: str
    relevance: RelevanceGrade
    notes: Optional[str] = None


# Persistence and Integrity Validation Utilities

def load_profiles_from_csv(file_path: Union[str, Path]) -> List[Profile]:
    """Load and validate helper profiles from a CSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Profiles file not found at {path}")

    profiles: List[Profile] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            try:
                profile = Profile(**row)
                profiles.append(profile)
            except Exception as e:
                raise ValueError(f"Failed to validate profile at row {row_idx}: {e}") from e
    return profiles


def save_profiles_to_csv(profiles: List[Profile], file_path: Union[str, Path]) -> None:
    """Serialize and write validated profiles to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "helper_id",
        "name",
        "skills",
        "bio",
        "languages",
        "availability",
        "session_mode",
        "location_area",
        "active",
    ]

    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in profiles:
            writer.writerow({
                "helper_id": p.helper_id,
                "name": p.name,
                "skills": "|".join(p.skills),
                "bio": p.bio,
                "languages": "|".join(p.languages),
                "availability": "|".join(p.availability),
                "session_mode": p.session_mode,
                "location_area": p.location_area or "",
                "active": str(p.active),
            })


def load_queries_from_json(file_path: Union[str, Path]) -> List[Query]:
    """Load and validate evaluation queries from JSON."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Queries file not found at {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Queries file must contain a JSON array of query objects, got {type(data)}")

    return [Query(**item) for item in data]


def save_queries_to_json(queries: List[Query], file_path: Union[str, Path]) -> None:
    """Serialize and write validated queries to JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode="w", encoding="utf-8") as f:
        json.dump([q.model_dump() for q in queries], f, indent=2, ensure_ascii=False)


def load_labels_from_json(file_path: Union[str, Path]) -> List[RelevanceLabel]:
    """Load and validate graded relevance labels from JSON."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found at {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Labels file must contain a JSON array, got {type(data)}")

    return [RelevanceLabel(**item) for item in data]


def save_labels_to_json(labels: List[RelevanceLabel], file_path: Union[str, Path]) -> None:
    """Serialize and write validated labels to JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode="w", encoding="utf-8") as f:
        json.dump([lbl.model_dump() for lbl in labels], f, indent=2, ensure_ascii=False)


def validate_corpus_integrity(
    profiles: List[Profile],
    queries: List[Query],
    labels: Optional[List[RelevanceLabel]] = None,
) -> List[str]:
    """Verify relational integrity across profiles, queries, and labels.

    Returns a list of error strings. If empty, the corpus is fully consistent.
    """
    errors: List[str] = []

    # 1. Unique IDs
    profile_ids: Set[str] = set()
    inactive_profile_ids: Set[str] = set()
    for p in profiles:
        if p.helper_id in profile_ids:
            errors.append(f"Duplicate helper_id found: '{p.helper_id}'")
        profile_ids.add(p.helper_id)
        if not p.active:
            inactive_profile_ids.add(p.helper_id)

    query_ids: Set[str] = set()
    for q in queries:
        if q.query_id in query_ids:
            errors.append(f"Duplicate query_id found: '{q.query_id}'")
        query_ids.add(q.query_id)

        # 2. Check all relevant_helper_ids exist in profiles
        for hid in q.relevant_helper_ids:
            if hid not in profile_ids:
                errors.append(
                    f"Query '{q.query_id}' references non-existent helper_id '{hid}'"
                )
            if hid in inactive_profile_ids:
                errors.append(
                    f"Query '{q.query_id}' references INACTIVE helper_id '{hid}'. Inactive profiles cannot be relevant."
                )

        # 3. No-match queries must have empty relevant_helper_ids
        if q.category == "no_match" and len(q.relevant_helper_ids) > 0:
            errors.append(
                f"No-match query '{q.query_id}' has non-empty relevant_helper_ids: {q.relevant_helper_ids}"
            )

    # 4. If labels are provided, check query_id and helper_id references
    if labels:
        for lbl in labels:
            if lbl.query_id not in query_ids:
                errors.append(f"Label references unknown query_id '{lbl.query_id}'")
            if lbl.helper_id not in profile_ids:
                errors.append(f"Label references unknown helper_id '{lbl.helper_id}'")

    return errors
