"""Automated tests for constraint evaluation in src/constraints.py."""

import pytest
from src.data_loader import Constraint, Profile
from src.constraints import (
    check_hard_constraints,
    filter_eligible_helpers,
    evaluate_preferences,
    is_availability_satisfied,
    is_language_satisfied,
    is_session_mode_satisfied,
)


@pytest.fixture
def sample_profiles():
    return [
        Profile(
            helper_id="H001",
            name="Harpreet K.",
            skills=["resume_review"],
            bio="CV review in Punjabi and English on Saturdays.",
            languages=["English", "Punjabi"],
            availability=["Saturday"],
            session_mode="online",
            location_area="North",
            active=True,
        ),
        Profile(
            helper_id="H002",
            name="David L.",
            skills=["guitar_lessons"],
            bio="Acoustic guitar in English on Sundays.",
            languages=["English"],
            availability=["Sunday"],
            session_mode="in_person",
            location_area="West",
            active=True,
        ),
        Profile(
            helper_id="H003",
            name="Inactive Punjabi Tutor",
            skills=["resume_review"],
            bio="Inactive CV tutor.",
            languages=["English", "Punjabi"],
            availability=["Saturday"],
            session_mode="online",
            location_area="North",
            active=False,  # Inactive
        ),
    ]


def test_language_satisfaction():
    assert is_language_satisfied(["English", "Punjabi"], "punjabi") is True
    assert is_language_satisfied(["English", "Punjabi"], "PUNJABI") is True
    assert is_language_satisfied(["English"], "Punjabi") is False


def test_availability_satisfaction():
    assert is_availability_satisfied(["Saturday", "Sunday"], "Saturday") is True
    assert is_availability_satisfied(["Saturday"], "weekend") is True
    assert is_availability_satisfied(["Weekend evenings"], "Saturday") is True
    assert is_availability_satisfied(["Weekday evenings"], "Saturday") is False


def test_session_mode_satisfaction():
    assert is_session_mode_satisfied("both", "online") is True
    assert is_session_mode_satisfied("online", "both") is True
    assert is_session_mode_satisfied("online", "online") is True
    assert is_session_mode_satisfied("in_person", "online") is False


def test_hard_constraints_filter(sample_profiles):
    # Require Punjabi language
    constraints = [Constraint(field="language", value="Punjabi", rule="required")]

    h1_eligible, _ = check_hard_constraints(sample_profiles[0], constraints)
    assert h1_eligible is True

    # H002 does not speak Punjabi
    h2_eligible, violations = check_hard_constraints(sample_profiles[1], constraints)
    assert h2_eligible is False
    assert any("language" in v for v in violations)

    # H003 speaks Punjabi but is INACTIVE -> must be disqualified!
    h3_eligible, violations = check_hard_constraints(sample_profiles[2], constraints)
    assert h3_eligible is False
    assert any("inactive" in v for v in violations)


def test_filter_eligible_helpers(sample_profiles):
    constraints = [
        Constraint(field="language", value="Punjabi", rule="required"),
        Constraint(field="availability", value="Saturday", rule="required"),
    ]
    eligible_ids = filter_eligible_helpers(sample_profiles, constraints)
    # Only H001 satisfies both AND is active
    assert eligible_ids == {"H001"}


def test_soft_preferences_do_not_disqualify(sample_profiles):
    # H002 only speaks English and is available Sunday.
    # Requester prefers Punjabi and Saturday.
    constraints = [
        Constraint(field="language", value="Punjabi", rule="preferred"),
        Constraint(field="availability", value="Saturday", rule="preferred"),
    ]

    # Hard constraints check passes because all constraints are preferred
    is_eligible, _ = check_hard_constraints(sample_profiles[1], constraints)
    assert is_eligible is True

    # But preference score reflects unmet preferences
    prefs = evaluate_preferences(sample_profiles[1], constraints)
    assert prefs["preference_score"] == 0.0
    assert len(prefs["unmet_preferences"]) == 2
    assert any("Punjabi" in label for label in prefs["unmet_preferences"])
    assert any("Saturday" in label for label in prefs["unmet_preferences"])


def test_soft_preferences_partial_match(sample_profiles):
    # H001 speaks Punjabi and is available Saturday.
    # Requester prefers Punjabi and Sunday.
    constraints = [
        Constraint(field="language", value="Punjabi", rule="preferred"),
        Constraint(field="availability", value="Sunday", rule="preferred"),
    ]

    prefs = evaluate_preferences(sample_profiles[0], constraints)
    # Meets 1 of 2 preferred constraints
    assert prefs["preference_score"] == 0.5
    assert prefs["language_fit"] == 1.0
    assert prefs["availability_fit"] == 0.0
    assert len(prefs["unmet_preferences"]) == 1
