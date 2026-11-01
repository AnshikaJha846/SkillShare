"""SkillShare Match: Streamlit Interactive Search & Recommendation Demo."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import streamlit as st
from src.constraints import filter_eligible_helpers, evaluate_preferences
from src.data_loader import Constraint, Profile, load_profiles_from_csv
from src.explain import generate_explanation
from src.retrieval import get_retriever


# Set page layout
st.set_page_config(
    page_title="SkillShare Match",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .helper-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .helper-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .score-badge {
        background-color: #3B82F6;
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .skill-tag {
        display: inline-block;
        background: #E0E7FF;
        color: #3730A3;
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .explanation-box {
        background: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 0.75rem;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #065F46;
        margin-top: 0.75rem;
    }
    .unmet-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.6rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 0.5rem;
    }
    .no-match-box {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 8px;
        padding: 1.5rem;
        color: #991B1B;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading profile corpus and caching vector models...")
def load_corpus_and_retrievers():
    profiles = load_profiles_from_csv("data/profiles.csv")
    return profiles


profiles = load_corpus_and_retrievers()

# Header
st.markdown('<div class="main-title">🤝 SkillShare Match</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'An explainable community skill-matching prototype. Matches by meaning, enforces '
    'mandatory constraints as <b>hard filters</b>, and explains recommendations with '
    '<b>zero hallucinations</b>.'
    '</div>',
    unsafe_allow_html=True,
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Matching Engine")
    retrieval_method = st.selectbox(
        "Retrieval Algorithm",
        options=[
            ("sbert_multi", "Multilingual S-BERT (MiniLM-L12)"),
            ("sbert_en", "English S-BERT (MiniLM-L6)"),
            ("bm25", "Okapi BM25 (Lexical)"),
            ("tfidf", "TF-IDF + Cosine (Baseline)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    st.markdown("---")
    st.subheader("💡 Try Sample Queries")
    sample_col1, sample_col2 = st.columns(2)

    sample_query = None
    sample_constraints = []

    if sample_col1.button("📄 Paraphrased CV"):
        sample_query = "Looking for someone to help improve my CV for upcoming software internships."
        sample_constraints = [
            Constraint(field="language", value="English", rule="preferred"),
            Constraint(field="availability", value="Saturday", rule="preferred"),
        ]

    if sample_col2.button("🌐 Gurmukhi Web"):
        sample_query = "ਮੈਨੂੰ ਵੈੱਬਸਾਈਟ ਬਣਾਉਣ ਲਈ HTML, CSS ਅਤੇ ਜਾਵਾਸਕ੍ਰਿਪਟ ਸਿੱਖਣੀ ਹੈ।"
        sample_constraints = [
            Constraint(field="language", value="Punjabi", rule="required"),
        ]

    if sample_col1.button("📊 Small Biz Excel"):
        sample_query = "I run a small shop and need help organizing my numbers and sales sheets better."
        sample_constraints = [
            Constraint(field="session_mode", value="online", rule="preferred"),
        ]

    if sample_col2.button("🚫 Scuba (No Match)"):
        sample_query = "Looking for certified scuba diving lessons and open water PADI training certification."
        sample_constraints = []

    st.markdown("---")
    st.caption("All profiles and requests are synthetic constructs for academic NLP research.")

# Main input form
with st.container():
    default_text = sample_query if sample_query else ""
    query_text = st.text_area(
        "What skill or help are you looking for?",
        value=default_text,
        height=100,
        placeholder="e.g., I need help improving my CV for internships, preferably weekends in Punjabi or English...",
        help="Type in natural language, whether formal, colloquial, or code-mixed Hindi/Punjabi.",
    )

    with st.expander("🎯 Practical Constraints & Filters (Language, Day, Mode)", expanded=bool(sample_constraints)):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            lang_val = st.selectbox(
                "Language",
                options=["Any", "English", "Punjabi", "Hindi", "French", "German"],
                index=2 if any(c.field == "language" and c.value == "Punjabi" for c in sample_constraints) else 0,
            )
            lang_rule = st.radio(
                "Language Mode",
                ["Required (Hard Filter)", "Preferred (Soft Bonus)"],
                index=0 if any(c.field == "language" and c.rule == "required" for c in sample_constraints) else 1,
                key="lang_rule",
            )

        with col2:
            avail_val = st.selectbox(
                "Availability",
                options=["Any", "Saturday", "Sunday", "Weekday evenings", "Weekday mornings", "Wednesday mornings"],
                index=1 if any(c.field == "availability" and c.value == "Saturday" for c in sample_constraints) else 0,
            )
            avail_rule = st.radio(
                "Availability Mode",
                ["Required (Hard Filter)", "Preferred (Soft Bonus)"],
                index=1,
                key="avail_rule",
            )

        with col3:
            mode_val = st.selectbox(
                "Session Mode",
                options=["Any", "online", "in_person", "both"],
                index=1 if any(c.field == "session_mode" and c.value == "online" for c in sample_constraints) else 0,
            )
            mode_rule = st.radio(
                "Session Mode Filter",
                ["Required (Hard Filter)", "Preferred (Soft Bonus)"],
                index=1,
                key="mode_rule",
            )

        with col4:
            loc_val = st.selectbox(
                "Location Area (In-person)",
                options=["Any", "North", "South", "Central", "East", "West"],
                index=0,
            )
            loc_rule = st.radio(
                "Location Filter",
                ["Required (Hard Filter)", "Preferred (Soft Bonus)"],
                index=1,
                key="loc_rule",
            )

    submit = st.button("🔍 Find Matching Helpers", type="primary", use_container_width=True)

if submit or (sample_query and not query_text):
    active_query = query_text.strip() if query_text.strip() else (sample_query or "")

    if not active_query:
        st.warning("Please type a description of what help you need.")
    else:
        # Build structured constraints
        active_constraints: List[Constraint] = []
        if lang_val != "Any":
            rule = "required" if "Required" in lang_rule else "preferred"
            active_constraints.append(Constraint(field="language", value=lang_val, rule=rule))
        if avail_val != "Any":
            rule = "required" if "Required" in avail_rule else "preferred"
            active_constraints.append(Constraint(field="availability", value=avail_val, rule=rule))
        if mode_val != "Any":
            rule = "required" if "Required" in mode_rule else "preferred"
            active_constraints.append(Constraint(field="session_mode", value=mode_val, rule=rule))
        if loc_val != "Any":
            rule = "required" if "Required" in loc_rule else "preferred"
            active_constraints.append(Constraint(field="location_area", value=loc_val, rule=rule))

        # 1. HARD CONSTRAINT FILTERING
        eligible_ids = filter_eligible_helpers(profiles, active_constraints)

        st.markdown(f"### Results ({len(eligible_ids)} active helpers meet mandatory constraints)")

        if not eligible_ids:
            st.markdown("""
            <div class="no-match-box">
                <h4>🚫 No eligible helpers found</h4>
                <p>Zero active profiles satisfied all of your <b>required</b> constraints (language, day, or location).</p>
                <p><b>Recommendation:</b> Try switching one or more constraints to <i>Preferred</i> rather than <i>Required</i>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Thresholds are calibrated per-method:
            # TF-IDF/BM25 produce low absolute cosine scores on small corpora (0.05 is already meaningful).
            # SBERT produces true cosine similarities in [0,1] so 0.30 filters weak semantic matches.
            _threshold_map = {
                "tfidf": 0.05,
                "bm25": 0.05,
                "sbert_en": 0.30,
                "sbert_multi": 0.30,
            }
            min_threshold = _threshold_map.get(retrieval_method, 0.05)

            with st.spinner("Ranking candidates..."):
                retriever = get_retriever(retrieval_method, profiles)
                results = retriever.search(
                    query_text=active_query,
                    top_k=3,
                    eligible_helper_ids=eligible_ids,
                    min_threshold=min_threshold,
                )

            if not results:
                st.markdown("""
                <div class="no-match-box">
                    <h4>🔍 No close skill matches found</h4>
                    <p>Although candidates passed your availability filters, none matched your skill request closely enough.</p>
                    <p><b>Suggestions:</b> Try describing your need using different words, or check if the requested skill is offered on the platform.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for idx, res in enumerate(results, start=1):
                    p = res.profile
                    # Evaluate soft preferences
                    pref_eval = evaluate_preferences(p, active_constraints)
                    unmet = pref_eval["unmet_preferences"]
                    explanation = generate_explanation(p, unmet_preferences=unmet)

                    skills_html = "".join([f'<span class="skill-tag">🏷️ {s.replace("_", " ")}</span>' for s in p.skills])

                    st.markdown(f"""
                    <div class="helper-card">
                        <div class="helper-header">
                            <div>
                                <h3 style="margin:0; color:#0F172A;">#{idx}. {p.name} <small style="color:#64748B; font-weight:normal;">({p.helper_id})</small></h3>
                                <p style="margin:0.2rem 0; color:#475569; font-size:0.9rem;">📍 {p.location_area or 'Online'} • 🗣️ {', '.join(p.languages)} • 🕒 {', '.join(p.availability)} • 💻 {p.session_mode}</p>
                            </div>
                            <div>
                                <span class="score-badge">Match Signal: {res.score:.2f}</span>
                            </div>
                        </div>
                        <div style="margin-bottom:0.6rem;">{skills_html}</div>
                        <p style="margin:0; font-style:italic; color:#334155; font-size:0.95rem;">"{p.bio}"</p>
                        <div class="explanation-box">
                            <b>💡 Grounded Explanation:</b> {explanation}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if unmet:
                        st.markdown(
                            f'<div class="unmet-box">⚠️ <b>Unmet Preference:</b> {"; ".join(unmet)}</div>',
                            unsafe_allow_html=True,
                        )
