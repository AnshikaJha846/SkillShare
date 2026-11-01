"""Automated tests for explanation generation and grounding checks in src/explain.py."""

import pytest
from src.data_loader import Profile
from src.explain import generate_explanation, verify_explanation_grounding


@pytest.fixture
def sample_profile():
    return Profile(
        helper_id="H001",
        name="Harpreet K.",
        skills=["resume_review", "interview_prep"],
        bio="HR professional offering CV reviews and mock interviews.",
        languages=["English", "Punjabi"],
        availability=["Saturday", "Sunday"],
        session_mode="both",
        location_area="North",
        active=True,
    )


def test_explanation_generation_and_grounding(sample_profile):
    exp = generate_explanation(sample_profile)

    assert "resume review" in exp
    assert "English" in exp or "Punjabi" in exp
    assert "Saturday" in exp or "Sunday" in exp

    is_grounded, errors = verify_explanation_grounding(exp, sample_profile)
    assert is_grounded is True
    assert errors == []


def test_explanation_with_unmet_preferences(sample_profile):
    unmet = ["Prefers Hindi (Helper offers: English, Punjabi)"]
    exp = generate_explanation(sample_profile, unmet_preferences=unmet)

    assert "Preference mismatch: Prefers Hindi" in exp
    is_grounded, _ = verify_explanation_grounding(exp, sample_profile)
    assert is_grounded is True


def test_grounding_audit_catches_untrue_claims(sample_profile):
    hallucinated_exp = "Recommended because this profile offers quantum physics tutoring in German."
    is_grounded, errors = verify_explanation_grounding(hallucinated_exp, sample_profile)

    assert is_grounded is False
    assert len(errors) > 0
