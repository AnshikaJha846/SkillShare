"""Retrieval models and ranking implementations for SkillShare Match.

Implements the TF-IDF + Cosine Similarity baseline with support for:
- Exclusion of inactive helper profiles.
- Candidate set filtering (hard constraints integration).
- Deterministic score tie-breaking.
- Safe handling of empty, whitespace, or out-of-vocabulary queries.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Set
import numpy as np

from pydantic import BaseModel, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import Profile
from src.preprocessing import extract_search_text, normalize_text


class RetrievalResult(BaseModel):
    """Container for a single retrieved helper candidate and ranking score."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    helper_id: str
    score: float
    profile: Profile



class TFIDFRetriever:
    """Lexical baseline retriever using TF-IDF sparse representations and Cosine Similarity."""

    def __init__(self, profiles: Sequence[Profile]) -> None:
        """Initialize the retriever, indexing ONLY active profiles."""
        # Hard Rule: Inactive profiles are strictly excluded from the index
        self.active_profiles: List[Profile] = [p for p in profiles if p.active]
        self.profile_map = {p.helper_id: p for p in self.active_profiles}

        # Token pattern retains words with 1+ characters, accents, and common tech symbols
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w\+#\.]+\b",
            sublinear_tf=True,
            ngram_range=(1, 2),
        )

        if self.active_profiles:
            corpus = [extract_search_text(p.skills, p.bio) for p in self.active_profiles]
            self.profile_matrix = self.vectorizer.fit_transform(corpus)
        else:
            self.profile_matrix = None

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        eligible_helper_ids: Optional[Set[str]] = None,
        min_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Rank active helper profiles against a query string.

        Args:
            query_text: Natural language user request.
            top_k: Maximum number of recommendations to return.
            eligible_helper_ids: Optional subset of helper IDs passing hard filters.
            min_threshold: Minimum similarity score required to be considered a match.

        Returns:
            List of RetrievalResult objects sorted descending by score.
        """
        # Empty or whitespace query handling: return empty list safely
        clean_query = normalize_text(query_text)
        if not clean_query or not self.active_profiles or self.profile_matrix is None:
            return []

        # Determine eligible candidate profile indices
        if eligible_helper_ids is not None:
            candidate_indices = [
                idx for idx, p in enumerate(self.active_profiles)
                if p.helper_id in eligible_helper_ids
            ]
            if not candidate_indices:
                return []
        else:
            candidate_indices = list(range(len(self.active_profiles)))

        # Vectorize query
        query_vec = self.vectorizer.transform([clean_query])

        # Compute cosine similarity
        candidate_matrix = self.profile_matrix[candidate_indices]
        sims = cosine_similarity(query_vec, candidate_matrix).flatten()

        # Build candidate results
        candidates: List[RetrievalResult] = []
        for local_idx, score in enumerate(sims):
            score_float = float(score)
            if score_float >= min_threshold:
                global_idx = candidate_indices[local_idx]
                profile = self.active_profiles[global_idx]
                candidates.append(
                    RetrievalResult(
                        helper_id=profile.helper_id,
                        score=round(score_float, 4),
                        profile=profile,
                    )
                )

        # Sort descending by score; break ties deterministically by helper_id
        candidates.sort(key=lambda item: (-item.score, item.helper_id))

        return candidates[:top_k]


class BM25Retriever:
    """Lexical baseline retriever using Okapi BM25 and term saturation scoring."""

    def __init__(self, profiles: Sequence[Profile]) -> None:
        """Initialize BM25 index over active profiles."""
        from rank_bm25 import BM25Okapi

        # Hard Rule: Inactive profiles are strictly excluded from the index
        self.active_profiles: List[Profile] = [p for p in profiles if p.active]
        self.profile_map = {p.helper_id: p for p in self.active_profiles}

        if self.active_profiles:
            self.tokenized_corpus = [
                self._tokenize(extract_search_text(p.skills, p.bio))
                for p in self.active_profiles
            ]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.tokenized_corpus = []
            self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Preservative token pattern keeping alphanumeric tokens, accents, and tech symbols."""
        clean = normalize_text(text)
        return re.findall(r"(?u)\b[\w\+#\.]+\b", clean)

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        eligible_helper_ids: Optional[Set[str]] = None,
        min_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Rank active helper profiles using BM25 scoring.

        Args:
            query_text: Natural language user request.
            top_k: Maximum number of recommendations to return.
            eligible_helper_ids: Optional subset of helper IDs passing hard filters.
            min_threshold: Minimum score required to be considered a match.

        Returns:
            List of RetrievalResult objects sorted descending by score.
        """
        clean_query = normalize_text(query_text)
        tokens = self._tokenize(clean_query)
        if not tokens or not self.active_profiles or self.bm25 is None:
            return []

        # Determine eligible candidate profile indices
        if eligible_helper_ids is not None:
            candidate_indices = [
                idx for idx, p in enumerate(self.active_profiles)
                if p.helper_id in eligible_helper_ids
            ]
            if not candidate_indices:
                return []
        else:
            candidate_indices = list(range(len(self.active_profiles)))

        # Get raw BM25 scores
        raw_scores = self.bm25.get_scores(tokens)

        # Normalize scores to 0-1 range for uniform ranking signals
        max_score = float(np.max(raw_scores)) if len(raw_scores) > 0 else 0.0

        candidates: List[RetrievalResult] = []
        for local_idx in candidate_indices:
            raw_s = float(raw_scores[local_idx])
            norm_s = (raw_s / max_score) if max_score > 0 else 0.0

            if norm_s >= min_threshold and raw_s > 0.0:
                profile = self.active_profiles[local_idx]
                candidates.append(
                    RetrievalResult(
                        helper_id=profile.helper_id,
                        score=round(norm_s, 4),
                        profile=profile,
                    )
                )

        # Sort descending by score; break ties deterministically by helper_id
        candidates.sort(key=lambda item: (-item.score, item.helper_id))

        return candidates[:top_k]


class DenseRetriever:
    """Semantic bi-encoder retriever using Sentence-BERT embeddings and Cosine Similarity."""

    def __init__(
        self,
        profiles: Sequence[Profile],
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Union[str, Path] = "data/cache",
        force_recompute: bool = False,
        device: Optional[str] = None,
    ) -> None:
        """Initialize dense index over active profiles, utilizing disk vector caching."""
        from sentence_transformers import SentenceTransformer
        from src.indexing import get_or_compute_profile_embeddings

        # Hard Rule: Inactive profiles are strictly excluded from the index
        self.active_profiles: List[Profile] = [p for p in profiles if p.active]
        self.profile_map = {p.helper_id: p for p in self.active_profiles}
        self.model_name = model_name
        self.device = device

        if self.active_profiles:
            self.model = SentenceTransformer(model_name, device=device)
            self.embeddings, self.indexed_ids = get_or_compute_profile_embeddings(
                profiles=self.active_profiles,
                model_name=model_name,
                cache_dir=cache_dir,
                force_recompute=force_recompute,
                device=device,
            )
            # Map helper_id to row index in self.embeddings
            self.id_to_idx = {hid: idx for idx, hid in enumerate(self.indexed_ids)}
        else:
            self.model = None
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.indexed_ids = []
            self.id_to_idx = {}

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        eligible_helper_ids: Optional[Set[str]] = None,
        min_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Rank active helper profiles using dense vector cosine similarity.

        Args:
            query_text: Natural language user request.
            top_k: Maximum number of recommendations to return.
            eligible_helper_ids: Optional subset of helper IDs passing hard filters.
            min_threshold: Minimum similarity score required to be considered a match.

        Returns:
            List of RetrievalResult objects sorted descending by score.
        """
        clean_query = normalize_text(query_text)
        if not clean_query or not self.active_profiles or self.model is None or len(self.embeddings) == 0:
            return []

        # Determine eligible row indices in embeddings matrix
        if eligible_helper_ids is not None:
            valid_hids = [hid for hid in self.indexed_ids if hid in eligible_helper_ids]
            if not valid_hids:
                return []
            candidate_row_indices = [self.id_to_idx[hid] for hid in valid_hids]
        else:
            candidate_row_indices = list(range(len(self.indexed_ids)))

        # Compute normalized query vector
        query_vector = self.model.encode(
            [clean_query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_vector = np.asarray(query_vector, dtype=np.float32)

        # Candidate matrix subset
        candidate_matrix = self.embeddings[candidate_row_indices]

        # Since vectors are unit-normalized, dot product is exact cosine similarity
        cosine_sims = np.dot(candidate_matrix, query_vector.T).flatten()

        candidates: List[RetrievalResult] = []
        for local_idx, score in enumerate(cosine_sims):
            score_float = float(score)
            # Clip negative similarities to 0.0
            score_clamped = max(0.0, min(1.0, score_float))

            if score_clamped >= min_threshold:
                global_idx = candidate_row_indices[local_idx]
                helper_id = self.indexed_ids[global_idx]
                profile = self.profile_map[helper_id]
                candidates.append(
                    RetrievalResult(
                        helper_id=helper_id,
                        score=round(score_clamped, 4),
                        profile=profile,
                    )
                )

        # Sort descending by score; break ties deterministically by helper_id
        candidates.sort(key=lambda item: (-item.score, item.helper_id))

        return candidates[:top_k]


def get_retriever(
    method: str,
    profiles: Sequence[Profile],
    cache_dir: Union[str, Path] = "data/cache",
    device: Optional[str] = None,
):
    """Factory creating the appropriate configured retriever for uniform comparison.

    Supported methods:
        - 'tfidf': TF-IDF baseline + cosine similarity
        - 'bm25': Okapi BM25 lexical baseline
        - 'sbert_en': English bi-encoder (all-MiniLM-L6-v2)
        - 'sbert_multi': Multilingual bi-encoder (paraphrase-multilingual-MiniLM-L12-v2)
    """
    clean_method = method.strip().lower()
    if clean_method == "tfidf":
        return TFIDFRetriever(profiles)
    elif clean_method == "bm25":
        return BM25Retriever(profiles)
    elif clean_method in {"sbert_en", "sbert", "dense_en"}:
        return DenseRetriever(
            profiles=profiles,
            model_name="all-MiniLM-L6-v2",
            cache_dir=cache_dir,
            device=device,
        )
    elif clean_method in {"sbert_multi", "multilingual", "dense_multi"}:
        return DenseRetriever(
            profiles=profiles,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir=cache_dir,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown retrieval method '{method}'. Supported methods: 'tfidf', 'bm25', 'sbert_en', 'sbert_multi'"
        )


