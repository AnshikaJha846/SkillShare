"""Unit tests for src/data_loader.py."""

import json
import pytest
from pydantic import ValidationError

from src.data_loader import (
    Constraint,
    Profile,
    Query,
    RelevanceLabel,
    load_profiles_from_csv,
    save_profiles_to_csv,
    load_queries_from_json,
    save_queries_to_json,
    load_labels_from_json,
    save_labels_to_json,
    validate_corpus_integrity,
)


def test_valid_profile_creation():
    profile = Profile(
        helper_id="H001",
        name="Harpreet K.",
        skills=["resume_review", "interview_prep"],
        bio="Experienced mentor offering CV reviews on weekends.",
        languages=["English", "Punjabi"],
        availability=["Saturday", "Sunday"],
        session_mode="online",
        location_area="North",
        active=True,
    )
    assert profile.helper_id == "H001"
    assert profile.name == "Harpreet K."
    assert "resume_review" in profile.skills
    assert profile.active is True


def test_profile_delimited_string_parsing():
    profile = Profile(
        helper_id="H002",
        name="Simran S.",
        skills="python | data_analysis | excel",
        bio="Data analyst willing to help beginners.",
        languages="Hindi, English",
        availability="Weekend evenings",
        session_mode="both",
        active="true",
    )
    assert profile.skills == ["python", "data_analysis", "excel"]
    assert profile.languages == ["Hindi", "English"]
    assert profile.availability == ["Weekend evenings"]
    assert profile.active is True


def test_profile_validation_empty_fields_fail():
    with pytest.raises(ValidationError):
        Profile(
            helper_id="",
            name="Test",
            skills=["python"],
            bio="Valid bio text here",
            languages=["English"],
            availability=["Monday"],
            session_mode="online",
        )

    with pytest.raises(ValidationError):
        Profile(
            helper_id="H003",
            name="Test",
            skills=[],  # empty skills list
            bio="Valid bio text here",
            languages=["English"],
            availability=["Monday"],
            session_mode="online",
        )


def test_constraint_validation():
    c1 = Constraint(field="language", value="Punjabi", rule="required")
    assert c1.rule == "required"
    assert c1.value == "Punjabi"

    c2 = Constraint(field="availability", value=["Saturday", "Sunday"], rule="preferred")
    assert c2.rule == "preferred"
    assert len(c2.value) == 2

    with pytest.raises(ValidationError):
        Constraint(field="language", value="   ")


def test_query_validation():
    q = Query(
        query_id="Q001",
        text="I need help improving my CV for internships",
        category="paraphrased",
        constraints=[Constraint(field="language", value="Punjabi", rule="required")],
        relevant_helper_ids=["H001"],
        split="validation",
    )
    assert q.query_id == "Q001"
    assert len(q.constraints) == 1
    assert q.relevant_helper_ids == ["H001"]


def test_query_no_match_cannot_have_relevant_helpers():
    with pytest.raises(ValidationError, match="no_match"):
        Query(
            query_id="Q002",
            text="Need help repairing a jet engine turbine",
            category="no_match",
            relevant_helper_ids=["H001"],  # invalid for no_match
            split="test",
        )


def test_profile_csv_roundtrip(tmp_path):
    csv_file = tmp_path / "profiles.csv"
    profiles = [
        Profile(
            helper_id="H001",
            name="Harpreet K.",
            skills=["resume_review", "interview_prep"],
            bio="Experienced mentor offering CV reviews on weekends.",
            languages=["English", "Punjabi"],
            availability=["Saturday", "Sunday"],
            session_mode="online",
            location_area="North",
            active=True,
        ),
        Profile(
            helper_id="H002",
            name="Raj V.",
            skills=["python", "excel"],
            bio="Junior developer helping with programming tasks.",
            languages=["English", "Hindi"],
            availability=["Weekday evenings"],
            session_mode="both",
            location_area="Central",
            active=False,
        ),
    ]

    save_profiles_to_csv(profiles, csv_file)
    loaded = load_profiles_from_csv(csv_file)

    assert len(loaded) == 2
    assert loaded[0].helper_id == "H001"
    assert loaded[0].skills == ["resume_review", "interview_prep"]
    assert loaded[0].active is True
    assert loaded[1].helper_id == "H002"
    assert loaded[1].active is False


def test_query_json_roundtrip(tmp_path):
    json_file = tmp_path / "queries.json"
    queries = [
        Query(
            query_id="Q001",
            text="CV help please",
            category="paraphrased",
            constraints=[Constraint(field="language", value="English", rule="preferred")],
            relevant_helper_ids=["H001"],
            split="validation",
        ),
        Query(
            query_id="Q002",
            text="Need commercial pilot lessons",
            category="no_match",
            constraints=[],
            relevant_helper_ids=[],
            split="test",
        ),
    ]

    save_queries_to_json(queries, json_file)
    loaded = load_queries_from_json(json_file)

    assert len(loaded) == 2
    assert loaded[0].query_id == "Q001"
    assert loaded[1].category == "no_match"
    assert loaded[1].relevant_helper_ids == []


def test_corpus_integrity_validation():
    profiles = [
        Profile(
            helper_id="H001",
            name="Harpreet K.",
            skills=["resume_review"],
            bio="Bio text here for Harpreet",
            languages=["English"],
            availability=["Saturday"],
            session_mode="online",
            active=True,
        ),
        Profile(
            helper_id="H002",
            name="Raj V.",
            skills=["python"],
            bio="Bio text here for Raj",
            languages=["Hindi"],
            availability=["Sunday"],
            session_mode="in_person",
            active=False,
        ),
    ]

    queries = [
        Query(
            query_id="Q001",
            text="CV review",
            category="paraphrased",
            relevant_helper_ids=["H001"],
            split="validation",
        ),
        Query(
            query_id="Q002",
            text="Python tutoring",
            category="paraphrased",
            relevant_helper_ids=["H002"],  # Inactive helper! Should trigger error
            split="test",
        ),
        Query(
            query_id="Q003",
            text="Guitar lessons",
            category="paraphrased",
            relevant_helper_ids=["H999"],  # Non-existent helper! Should trigger error
            split="test",
        ),
    ]

    errors = validate_corpus_integrity(profiles, queries)
    assert len(errors) == 2
    assert any("INACTIVE" in e for e in errors)
    assert any("non-existent" in e for e in errors)


def test_load_actual_sample_dataset():
    profiles = load_profiles_from_csv("data/profiles.csv")
    assert len(profiles) == 120
    queries = load_queries_from_json("data/queries.json")
    assert len(queries) == 50
    labels = load_labels_from_json("data/labels.json")
    assert len(labels) > 0
    errors = validate_corpus_integrity(profiles, queries, labels)
    assert errors == []


