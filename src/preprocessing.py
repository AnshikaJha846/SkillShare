"""Text preprocessing utilities for SkillShare Match.

Preserves critical domain vocabulary, accents (e.g. résumé), numbers, and technical terms.
Avoids destructive regexes or aggressive stopword stripping that discard informative clues.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize text while preserving accents, abbreviations, and domain terms.

    - Normalizes unicode to NFC form so characters like 'é' are consistent.
    - Normalizes diverse whitespace (newlines, tabs) to single spaces.
    - Preserves meaningful symbols in terms like C++, C#, .NET, scikit-learn.
    - Avoids aggressive stripping of short words (e.g. CV, ML, AI, R, Go).
    """
    if not text:
        return ""

    # Unicode canonical composition
    normalized = unicodedata.normalize("NFC", text)

    # Normalize excessive whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if lowercase:
        normalized = normalized.lower()

    return normalized


def extract_search_text(skills: list[str], bio: str) -> str:
    """Construct an information-rich representation of a helper profile.

    Focuses strictly on capability and skill evidence to avoid spurious matches
    on practical constraint words like 'online' or 'Sunday'.
    """
    skill_text = " ".join([s.replace("_", " ") for s in skills])
    combined = f"{skill_text} {bio}"
    return normalize_text(combined, lowercase=True)
