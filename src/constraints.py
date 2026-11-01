"""Constraint evaluation and filtering engine for SkillShare Match.

Implements:
1. Hard Filters: Strictly removes candidates violating any required constraint
   (language, availability, session mode, location, or active status).
2. Soft Preferences: Computes transparent component scores for preferred constraints
   and flags unmet preferences for grounded UI feedback.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple
from src.data_loader import Constraint, Profile


def is_language_satisfied(helper_languages: Sequence[str], target_language: str) -> bool:
    """Case-insensitive check whether target language is spoken by the helper."""
    target_clean = target_language.strip().lower()
    return any(lang.strip().lower() == target_clean for lang in helper_languages)


def is_availability_satisfied(helper_availability: Sequence[str], target_availability: str) -> bool:
    """Check if helper availability overlaps with target day/window."""
    target_clean = target_availability.strip().lower()
    for slot in helper_availability:
        slot_clean = slot.strip().lower()
        if target_clean in slot_clean or slot_clean in target_clean:
            return True
        # Handle "weekends" matching Saturday or Sunday
        if target_clean in {"weekend", "weekends"} and ("saturday" in slot_clean or "sunday" in slot_clean):
            return True
        if "weekend" in slot_clean and target_clean in {"saturday", "sunday"}:
            return True
        # Handle "weekdays" matching Monday-Friday or weekday
        if target_clean in {"weekday", "weekdays"} and "weekday" in slot_clean:
            return True
    return False


def is_session_mode_satisfied(helper_mode: str, target_mode: str) -> bool:
    """Check whether helper session mode satisfies target requirement."""
    target_clean = target_mode.strip().lower()
    helper_clean = helper_mode.strip().lower()

    if target_clean in {"either", "both", "any"}:
        return True
    if helper_clean in {"both", "either"}:
        return True
    return target_clean == helper_clean


def is_location_satisfied(helper_location: Optional[str], target_location: str) -> bool:
    """Check broad location match for in-person requests."""
    if not target_location or target_location.strip().lower() in {"any", "online"}:
        return True
    if not helper_location:
        return False
    return target_location.strip().lower() in helper_location.strip().lower()


def check_hard_constraints(profile: Profile, constraints: Sequence[Constraint]) -> Tuple[bool, List[str]]:
    """Determine if a profile satisfies 100% of required constraints.

    Returns:
        (is_eligible, violation_reasons)
    """
    # Absolute invariant: inactive profiles are strictly ineligible
    if not profile.active:
        return False, ["Profile is inactive"]

    violations: List[str] = []
    for c in constraints:
        if c.rule != "required":
            continue

        field = c.field
        values = [c.value] if isinstance(c.value, str) else c.value

        for val in values:
            if field == "language":
                if not is_language_satisfied(profile.languages, val):
                    violations.append(f"Requires language '{val}', helper offers {profile.languages}")
            elif field == "availability":
                if not is_availability_satisfied(profile.availability, val):
                    violations.append(f"Requires availability '{val}', helper offers {profile.availability}")
            elif field == "session_mode":
                if not is_session_mode_satisfied(profile.session_mode, val):
                    violations.append(f"Requires session mode '{val}', helper offers '{profile.session_mode}'")
            elif field == "location_area":
                if not is_location_satisfied(profile.location_area, val):
                    violations.append(f"Requires area '{val}', helper in '{profile.location_area}'")

    return (len(violations) == 0, violations)


def filter_eligible_helpers(
    profiles: Sequence[Profile],
    constraints: Sequence[Constraint],
) -> Set[str]:
    """Filter a corpus of profiles to only those satisfying all hard constraints."""
    eligible_ids: Set[str] = set()
    for p in profiles:
        passed, _ = check_hard_constraints(p, constraints)
        if passed:
            eligible_ids.add(p.helper_id)
    return eligible_ids


def evaluate_preferences(
    profile: Profile,
    constraints: Sequence[Constraint],
) -> Dict[str, any]:
    """Evaluate preferred (soft) constraints, returning component scores and unmet labels."""
    preferred_constraints = [c for c in constraints if c.rule == "preferred"]

    if not preferred_constraints:
        return {
            "preference_score": 1.0,
            "language_fit": 1.0,
            "availability_fit": 1.0,
            "session_mode_fit": 1.0,
            "unmet_preferences": [],
        }

    met_count = 0
    total_count = len(preferred_constraints)
    unmet_labels: List[str] = []

    language_fit = 1.0
    availability_fit = 1.0
    session_mode_fit = 1.0

    for c in preferred_constraints:
        field = c.field
        values = [c.value] if isinstance(c.value, str) else c.value
        all_val_met = True

        for val in values:
            if field == "language":
                if is_language_satisfied(profile.languages, val):
                    met_count += 1
                else:
                    all_val_met = False
                    language_fit = 0.0
                    unmet_labels.append(f"Prefers {val} (Helper offers: {', '.join(profile.languages)})")
            elif field == "availability":
                if is_availability_satisfied(profile.availability, val):
                    met_count += 1
                else:
                    all_val_met = False
                    availability_fit = 0.0
                    unmet_labels.append(f"Prefers {val} (Helper available: {', '.join(profile.availability)})")
            elif field == "session_mode":
                if is_session_mode_satisfied(profile.session_mode, val):
                    met_count += 1
                else:
                    all_val_met = False
                    session_mode_fit = 0.0
                    unmet_labels.append(f"Prefers {val} (Helper offers: {profile.session_mode})")
            elif field == "location_area":
                if is_location_satisfied(profile.location_area, val):
                    met_count += 1
                else:
                    all_val_met = False
                    unmet_labels.append(f"Prefers {val} area (Helper in: {profile.location_area or 'unspecified'})")

    preference_score = round(met_count / total_count, 4) if total_count > 0 else 1.0

    return {
        "preference_score": preference_score,
        "language_fit": language_fit,
        "availability_fit": availability_fit,
        "session_mode_fit": session_mode_fit,
        "unmet_preferences": unmet_labels,
    }
