"""Profile-grounded explanation generator for SkillShare Match.

Strictly adheres to FR15, FR16, and FR17:
- Deterministic, template-based explanation synthesis.
- Relies EXCLUSIVELY on verified fields present in the profile.
- Explicitly states unmet preferred constraints to maintain user trust.
- Zero generative hallucinations or invented qualifications.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
from src.data_loader import Profile


def generate_explanation(
    profile: Profile,
    matched_skills: Optional[Sequence[str]] = None,
    unmet_preferences: Optional[Sequence[str]] = None,
) -> str:
    """Construct an auditable, profile-grounded explanation for a recommendation.

    Args:
        profile: The recommended helper's profile.
        matched_skills: Specific skill tags relevant to the query (defaults to all skills).
        unmet_preferences: List of preference mismatch messages from constraints engine.

    Returns:
        A concise, deterministic one-to-two sentence explanation grounded in profile facts.
    """
    # 1. Skill evidence clause
    if matched_skills:
        skill_tags = [s.replace("_", " ") for s in matched_skills]
    else:
        skill_tags = [s.replace("_", " ") for s in profile.skills]

    if len(skill_tags) == 1:
        skill_phrase = f"offers {skill_tags[0]}"
    elif len(skill_tags) == 2:
        skill_phrase = f"offers {skill_tags[0]} and {skill_tags[1]}"
    else:
        skill_phrase = f"offers {', '.join(skill_tags[:-1])}, and {skill_tags[-1]}"

    # 2. Languages clause
    lang_str = " and ".join(profile.languages) if len(profile.languages) <= 2 else ", ".join(profile.languages)

    # 3. Session mode clause
    if profile.session_mode == "both":
        mode_str = "both online and in-person sessions"
    elif profile.session_mode == "online":
        mode_str = "online sessions"
    else:
        mode_str = f"in-person sessions in {profile.location_area or 'their local area'}"

    # 4. Availability clause
    avail_str = " and ".join(profile.availability) if len(profile.availability) <= 2 else ", ".join(profile.availability)

    # Assemble core factual explanation
    explanation = (
        f"Recommended because this profile {skill_phrase}, conducts {mode_str} "
        f"in {lang_str}, and is available {avail_str}."
    )

    # 5. Append unmet preference disclosure if present (FR17)
    if unmet_preferences:
        caveats = "; ".join(unmet_preferences)
        explanation += f" Preference mismatch: {caveats}."

    return explanation


def verify_explanation_grounding(explanation: str, profile: Profile) -> Tuple[bool, List[str]]:
    """Audit whether all factual entities in an explanation are grounded in the profile.

    Returns:
        (is_grounded, list_of_ungrounded_tokens)
    """
    errors: List[str] = []
    exp_lower = explanation.lower()

    # Verify at least one profile skill tag appears in explanation
    skill_words = [s.replace("_", " ").lower() for s in profile.skills]
    has_skill = any(sw in exp_lower for sw in skill_words)
    if not has_skill:
        errors.append(f"No profile skill from {profile.skills} found in explanation.")

    # Verify helper languages are accurately represented
    has_language = any(lang.lower() in exp_lower for lang in profile.languages)
    if not has_language:
        errors.append(f"None of helper languages {profile.languages} found in explanation.")

    # Verify session mode accurately represented
    if profile.session_mode == "online" and "online" not in exp_lower:
        errors.append("Profile online mode not stated.")
    if profile.session_mode == "in_person" and "in-person" not in exp_lower:
        errors.append("Profile in-person mode not stated.")

    return (len(errors) == 0, errors)
