"""Automated tests for vector indexing, caching, and DenseRetriever."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.data_loader import Profile
from src.indexing import get_or_compute_profile_embeddings
from src.retrieval import DenseRetriever, get_retriever


@pytest.fixture
def sample_profiles():
    return [
        Profile(
            helper_id="H001",
            name="Harpreet K.",
            skills=["resume_review"],
            bio="Experienced HR professional offering resume reviews.",
            languages=["English", "Punjabi"],
            availability=["Saturday"],
            session_mode="online",
            active=True,
        ),
        Profile(
            helper_id="H002",
            name="David L.",
            skills=["guitar_lessons"],
            bio="Acoustic guitar instructor teaching chords.",
            languages=["English"],
            availability=["Sunday"],
            session_mode="in_person",
            active=True,
        ),
        Profile(
            helper_id="H003",
            name="Inactive Profile",
            skills=["resume_review"],
            bio="Inactive CV reviewer.",
            languages=["English"],
            availability=["Saturday"],
            session_mode="online",
            active=False,
        ),
    ]


def test_embedding_caching_and_inactive_exclusion(tmp_path, sample_profiles):
    """Verify caching saves to disk and inactive profiles are excluded."""
    # Mock SentenceTransformer encode to avoid downloading weights during unit testing
    mock_model = MagicMock()
    # 2 active profiles -> return shape (2, 8) normalized vectors
    dummy_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    mock_model.encode.return_value = dummy_vecs

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embs, ids = get_or_compute_profile_embeddings(
            profiles=sample_profiles,
            model_name="test-model",
            cache_dir=tmp_path,
        )

        assert ids == ["H001", "H002"]
        assert "H003" not in ids
        assert embs.shape == (2, 8)
        assert mock_model.encode.call_count == 1

        # Second call should load directly from disk cache without invoking model.encode again
        embs2, ids2 = get_or_compute_profile_embeddings(
            profiles=sample_profiles,
            model_name="test-model",
            cache_dir=tmp_path,
        )
        assert ids2 == ["H001", "H002"]
        assert np.array_equal(embs, embs2)
        # Call count remains 1 because disk cache was hit!
        assert mock_model.encode.call_count == 1


def test_dense_retriever_search_and_filtering(tmp_path, sample_profiles):
    """Verify DenseRetriever ranking, inactive exclusion, and candidate filtering."""
    mock_model = MagicMock()
    # H001 matches query vector [1, 0, 0, ...]
    dummy_profile_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],  # H001
        [0.0, 1.0, 0.0, 0.0],  # H002
    ], dtype=np.float32)
    mock_model.encode.side_effect = [
        dummy_profile_vecs,                      # Corpus encoding
        np.array([[1.0, 0.0, 0.0, 0.0]]),        # Query 1 (matches H001)
        np.array([[0.0, 1.0, 0.0, 0.0]]),        # Query 2 (matches H002)
    ]

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        retriever = DenseRetriever(
            profiles=sample_profiles,
            model_name="test-model",
            cache_dir=tmp_path,
        )

        # Search query matching H001
        results = retriever.search("I need a resume review", top_k=2)
        assert len(results) == 2
        assert results[0].helper_id == "H001"
        assert results[0].score == 1.0
        assert results[1].helper_id == "H002"
        assert results[1].score == 0.0

        # Inactive profile H003 must not be returned
        assert all(r.helper_id != "H003" for r in results)

        # Candidate filtering: Restrict candidates to only H002
        filtered = retriever.search("Resume review", eligible_helper_ids={"H002"})
        assert len(filtered) == 1
        assert filtered[0].helper_id == "H002"

        # Empty query handling
        assert retriever.search("") == []
        assert retriever.search("   ") == []


def test_retriever_factory(sample_profiles):
    """Verify get_retriever factory initializes all supported methods."""
    r_tfidf = get_retriever("tfidf", sample_profiles)
    assert r_tfidf.__class__.__name__ == "TFIDFRetriever"

    r_bm25 = get_retriever("bm25", sample_profiles)
    assert r_bm25.__class__.__name__ == "BM25Retriever"

    with pytest.raises(ValueError, match="Unknown retrieval method"):
        get_retriever("unsupported_method", sample_profiles)
