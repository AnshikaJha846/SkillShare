"""Embedding index management and disk caching for SkillShare Match.

Precomputes and caches dense vectors for active profiles to guarantee
sub-second offline search response times without repetitive model forward passes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from src.data_loader import Profile
from src.preprocessing import extract_search_text


def _sanitize_model_name(model_name: str) -> str:
    """Sanitize model identifier into a safe filename string."""
    return re.sub(r"[^\w\-_\.]", "_", model_name)


def get_or_compute_profile_embeddings(
    profiles: Sequence[Profile],
    model_name: str,
    cache_dir: Union[str, Path] = "data/cache",
    force_recompute: bool = False,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Retrieve cached dense embeddings or compute and persist them if missing.

    Only active profiles are indexed.

    Args:
        profiles: Full collection of helper profiles.
        model_name: Sentence-Transformers model name or path.
        cache_dir: Directory where embedding matrices are stored.
        force_recompute: If True, bypass cache and recompute.
        device: PyTorch device ('cpu', 'cuda', etc.).

    Returns:
        Tuple of (embeddings_matrix [N, D], list of N helper_ids).
    """
    from sentence_transformers import SentenceTransformer

    active_profiles: List[Profile] = [p for p in profiles if p.active]
    if not active_profiles:
        return np.empty((0, 0), dtype=np.float32), []

    active_ids = [p.helper_id for p in active_profiles]
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    sanitized_name = _sanitize_model_name(model_name)
    cache_file = cache_path / f"embeddings_{sanitized_name}_{len(active_profiles)}.npz"

    # Attempt to load from cache
    if not force_recompute and cache_file.is_file():
        try:
            cached_data = np.load(cache_file, allow_pickle=True)
            cached_ids = list(cached_data["helper_ids"])
            if cached_ids == active_ids:
                return cached_data["embeddings"].astype(np.float32), cached_ids
        except Exception:
            # If cache corrupted, fall through to recompute
            pass

    # Extract search texts for active profiles
    corpus_texts = [extract_search_text(p.skills, p.bio) for p in active_profiles]

    # Load model and compute embeddings
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        corpus_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Persist to disk for rapid offline retrieval
    np.savez_compressed(
        cache_file,
        embeddings=embeddings,
        helper_ids=np.array(active_ids, dtype=object),
    )

    return embeddings, active_ids
