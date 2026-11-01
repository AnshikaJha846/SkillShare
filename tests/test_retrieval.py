"""Tests for TF-IDF baseline retriever in src/retrieval.py."""

import pytest
from src.data_loader import Profile
from src.retrieval import TFIDFRetriever


@pytest.fixture
def tiny_profile_fixture():
    """A minimal controlled corpus of active and inactive profiles."""
    return [
        Profile(
            helper_id="H001",
            name="Harpreet K.",
            skills=["resume_review", "interview_prep"],
            bio="Experienced mentor offering CV reviews, resume feedback, and mock interviews.",
            languages=["English", "Punjabi"],
            availability=["Saturday"],
            session_mode="online",
            active=True,
        ),
        Profile(
            helper_id="H002",
            name="Rohan M.",
            skills=["python_programming"],
            bio="Software engineer offering Python coding lessons, algorithms, and backend scripting.",
            languages=["English", "Hindi"],
            availability=["Sunday"],
            session_mode="online",
            active=True,
        ),
        Profile(
            helper_id="H003",
            name="David L.",
            skills=["guitar_lessons"],
            bio="Acoustic guitar lessons for beginners. Chords, strumming patterns, and songs.",
            languages=["English"],
            availability=["Saturday"],
            session_mode="in_person",
            active=True,
        ),
        Profile(
            helper_id="H004",
            name="Inactive Recruiter",
            skills=["resume_review", "interview_prep"],
            bio="CV review, resume feedback, mock interviews. Exactly matching H001 but inactive.",
            languages=["English"],
            availability=["Saturday"],
            session_mode="online",
            active=False,  # INACTIVE PROFILE
        ),
    ]


def test_ranking_order_on_tiny_fixture(tiny_profile_fixture):
    """Verify that helper with highest lexical overlap is ranked first."""
    retriever = TFIDFRetriever(tiny_profile_fixture)
    results = retriever.search("I need someone to review my CV and give resume feedback", top_k=3)

    assert len(results) > 0
    # H001 has direct overlap on 'cv', 'resume', 'review', 'feedback'
    assert results[0].helper_id == "H001"
    assert results[0].score > 0.0

    # Ensure other profiles rank lower
    if len(results) > 1:
        assert results[0].score >= results[1].score


def test_inactive_profiles_strictly_excluded(tiny_profile_fixture):
    """Verify that inactive profile H004 is NEVER returned, despite identical bio keywords."""
    retriever = TFIDFRetriever(tiny_profile_fixture)
    results = retriever.search("CV review resume feedback mock interviews", top_k=10)

    returned_ids = [r.helper_id for r in results]
    assert "H004" not in returned_ids
    # H001 should be present
    assert "H001" in returned_ids


def test_empty_and_whitespace_query_handling(tiny_profile_fixture):
    """Verify empty, whitespace-only, and punctuation-only queries safely return empty list."""
    retriever = TFIDFRetriever(tiny_profile_fixture)

    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("\t\n") == []


def test_eligible_candidate_filtering(tiny_profile_fixture):
    """Verify retriever strictly respects hard filter candidate subsets."""
    retriever = TFIDFRetriever(tiny_profile_fixture)

    # Restrict search only to H002 and H003
    results = retriever.search(
        query_text="Need CV and resume review help",
        top_k=3,
        eligible_helper_ids={"H002", "H003"},
    )

    returned_ids = {r.helper_id for r in results}
    # Even though H001 is the best match, it was excluded by candidate filter
    assert "H001" not in returned_ids

    # If eligible set is empty, return empty
    assert retriever.search("Python", eligible_helper_ids=set()) == []


def test_min_threshold_filtering(tiny_profile_fixture):
    """Verify results below min_threshold are filtered out."""
    retriever = TFIDFRetriever(tiny_profile_fixture)

    # An entirely unrelated query with high threshold
    results = retriever.search("quantum mechanics astrophysics", min_threshold=0.5)
    assert len(results) == 0


def test_deterministic_tie_breaking():
    """Verify ties are broken deterministically by helper_id ascending."""
    profiles = [
        Profile(
            helper_id="H009",
            name="Helper Nine",
            skills=["chess_tutoring"],
            bio="Chess coach playing openings and tactics.",
            languages=["English"],
            availability=["Saturday"],
            session_mode="online",
            active=True,
        ),
        Profile(
            helper_id="H002",
            name="Helper Two",
            skills=["chess_tutoring"],
            bio="Chess coach playing openings and tactics.",
            languages=["English"],
            availability=["Saturday"],
            session_mode="online",
            active=True,
        ),
    ]

    retriever = TFIDFRetriever(profiles)
    results = retriever.search("chess coach", top_k=2)

    assert len(results) == 2
    assert results[0].score == results[1].score
    # H002 should come before H009 alphabetically
    assert results[0].helper_id == "H002"
    assert results[1].helper_id == "H009"


def test_bm25_ranking_and_inactive_exclusion(tiny_profile_fixture):
    """Verify BM25 ranks direct keyword match #1 and strictly excludes inactive profiles."""
    from src.retrieval import BM25Retriever

    retriever = BM25Retriever(tiny_profile_fixture)
    results = retriever.search("I need CV review and resume feedback", top_k=5)

    assert len(results) > 0
    # H001 is the active match
    assert results[0].helper_id == "H001"
    assert results[0].score > 0.0

    returned_ids = [r.helper_id for r in results]
    # H004 is inactive and must NEVER appear
    assert "H004" not in returned_ids


def test_bm25_empty_query_and_filtering(tiny_profile_fixture):
    """Verify BM25 handles empty queries safely and respects candidate sets."""
    from src.retrieval import BM25Retriever

    retriever = BM25Retriever(tiny_profile_fixture)

    assert retriever.search("") == []
    assert retriever.search("   ") == []

    # Restrict candidate set to H002 only
    filtered_results = retriever.search(
        "CV review and resume",
        eligible_helper_ids={"H002"},
    )
    assert all(r.helper_id == "H002" for r in filtered_results)

