"""SkillShare Match: Evaluation & Benchmark Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import pandas as pd
import streamlit as st


from src.constraints import filter_eligible_helpers
from src.data_loader import load_profiles_from_csv, load_queries_from_json
from src.explain import generate_explanation
from src.retrieval import get_retriever

st.set_page_config(
    page_title="Evaluation Dashboard - SkillShare Match",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Capstone Evaluation & Benchmark Dashboard")
st.markdown(
    "Rigorous experimental comparison of lexical baselines (TF-IDF, BM25) vs. "
    "dense semantic bi-encoders across **50 evaluation queries** and **120 helper profiles**."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Headline Benchmark Results",
    "🔍 Per-Category Breakdown",
    "🧪 Constraint Ablation Study",
    "🔬 Side-by-Side Model Comparison",
])

with tab1:
    st.subheader("Official Held-Out Test Split Metrics (25 Held-Out Queries)")
    st.markdown(
        "Headline metrics are computed with **95% non-parametric bootstrap confidence intervals** "
        "($N=1,000$ resamples) on identical held-out queries with mandatory hard constraint filtering."
    )

    metrics_df = pd.DataFrame([
        {
            "Retrieval Method": "1. TF-IDF + Cosine (Baseline)",
            "Precision@3": "0.493 [0.360, 0.653]",
            "Recall@3": "0.483 [0.345, 0.645]",
            "MRR": "0.613 [0.453, 0.780]",
            "NDCG@3": "0.550 [0.405, 0.714]",
            "Constraint Sat.": "100.0%",
            "No-Match Acc.": "100.0%",
        },
        {
            "Retrieval Method": "2. Okapi BM25 (Lexical)",
            "Precision@3": "0.573 [0.440, 0.707]",
            "Recall@3": "0.511 [0.378, 0.647]",
            "MRR": "0.673 [0.540, 0.813]",
            "NDCG@3": "0.610 [0.477, 0.753]",
            "Constraint Sat.": "100.0%",
            "No-Match Acc.": "92.0%",
        },
        {
            "Retrieval Method": "3. S-BERT English (all-MiniLM-L6-v2)",
            "Precision@3": "0.693 [0.520, 0.840]",
            "Recall@3": "0.580 [0.438, 0.711]",
            "MRR": "0.800 [0.640, 0.960]",
            "NDCG@3": "0.778 [0.603, 0.926]",
            "Constraint Sat.": "100.0%",
            "No-Match Acc.": "92.0%",
        },
        {
            "Retrieval Method": "4. Multilingual S-BERT (MiniLM-L12)",
            "Precision@3": "0.640 [0.453, 0.813]",
            "Recall@3": "0.494 [0.352, 0.632]",
            "MRR": "0.687 [0.506, 0.840]",
            "NDCG@3": "0.678 [0.493, 0.840]",
            "Constraint Sat.": "100.0%",
            "No-Match Acc.": "92.0%",
        },
    ])

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("""
    > **Key Finding**: S-BERT English achieves the highest overall NDCG@3 (`0.778`), outperforming the TF-IDF baseline (`0.550`) by **+22.8 points** on complex queries.
    """)

with tab2:
    st.subheader("Category-by-Category NDCG@3 Comparison")
    st.markdown(
        "Demonstrates where semantic models excel over lexical baselines across the 5 query archetypes."
    )

    cat_df = pd.DataFrame([
        {"Category": "Paraphrased (Vocabulary Mismatches)", "TF-IDF": 0.753, "BM25": 0.816, "S-BERT (EN)": 1.000, "S-BERT (Multi)": 1.000},
        {"Category": "Vague / Underspecified Requests", "TF-IDF": 0.106, "BM25": 0.378, "S-BERT (EN)": 1.000, "S-BERT (Multi)": 1.000},
        {"Category": "Multi-Skill Demands", "TF-IDF": 0.694, "BM25": 0.706, "S-BERT (EN)": 0.723, "S-BERT (Multi)": 0.524},
        {"Category": "Code-Mixed (Hindi/Punjabi)", "TF-IDF": 0.198, "BM25": 0.551, "S-BERT (EN)": 0.569, "S-BERT (Multi)": 0.267},
        {"Category": "No-Match Thresholding", "TF-IDF": 1.000, "BM25": 0.600, "S-BERT (EN)": 0.600, "S-BERT (Multi)": 0.600},
    ])

    st.dataframe(cat_df, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Observations:
    1. **Paraphrased Queries**: Semantic models achieve perfect NDCG@3 (`1.000`) resolving terms like *'CV'* vs. *'résumé'* and *'learn to program'* vs. *'Python scripting'*.
    2. **Vague Queries**: Lexical models fail severely (`0.106`), while dense models map broad needs (e.g., small business sales sheets) to relevant capabilities (`1.000`).
    """)

with tab3:
    st.subheader("Constraint Filtering Ablation Study")
    st.markdown(
        "Ablation comparing recommendation quality and constraint compliance with and without mandatory hard filters."
    )

    ablation_df = pd.DataFrame([
        {
            "Pipeline Configuration": "Multilingual S-BERT + Hard Constraint Filters (Default)",
            "Precision@3": "0.640",
            "Recall@3": "0.494",
            "NDCG@3": "0.678",
            "Constraint Satisfaction": "100.0%",
            "Violations Returned": "0 (Guaranteed)",
        },
        {
            "Pipeline Configuration": "Multilingual S-BERT WITHOUT Filters (Unconstrained)",
            "Precision@3": "0.507",
            "Recall@3": "0.368",
            "NDCG@3": "0.589",
            "Constraint Satisfaction": "86.7%",
            "Violations Returned": "Frequent language & day mismatch",
        },
    ])

    st.dataframe(ablation_df, use_container_width=True, hide_index=True)

    st.error(
        "**Core Architectural Lesson**: Without hard constraint filters, semantic models frequently "
        "recommend semantically similar helpers who speak the wrong language or are unavailable, "
        "causing practical user failure."
    )

with tab4:
    st.subheader("Interactive Side-by-Side Model Inspector")
    st.markdown("Enter any evaluation query or custom text to observe rankings from all 4 models simultaneously.")

    profiles = load_profiles_from_csv("data/profiles.csv")
    queries = load_queries_from_json("data/queries.json")

    selected_query_obj = st.selectbox(
        "Select a pre-labeled evaluation query to inspect:",
        options=queries,
        format_func=lambda q: f"[{q.category.upper()}] ({q.split}) {q.text[:80]}...",
    )

    if selected_query_obj:
        eligible_ids = filter_eligible_helpers(profiles, selected_query_obj.constraints)
        st.write(f"**Ground Truth Relevant Helper IDs**: `{selected_query_obj.relevant_helper_ids}`")
        st.write(f"**Eligible Candidates Passing Hard Constraints**: `{len(eligible_ids)} profiles`")

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.markdown("#### TF-IDF")
            r = get_retriever("tfidf", profiles)
            results = r.search(selected_query_obj.text, top_k=3, eligible_helper_ids=eligible_ids)
            for res in results:
                is_hit = "🎯 Relevant" if res.helper_id in selected_query_obj.relevant_helper_ids else "❌ Non-relevant"
                st.info(f"**{res.profile.name}** (`{res.helper_id}`)\nScore: {res.score:.2f} | {is_hit}")

        with col_b:
            st.markdown("#### BM25")
            r = get_retriever("bm25", profiles)
            results = r.search(selected_query_obj.text, top_k=3, eligible_helper_ids=eligible_ids)
            for res in results:
                is_hit = "🎯 Relevant" if res.helper_id in selected_query_obj.relevant_helper_ids else "❌ Non-relevant"
                st.info(f"**{res.profile.name}** (`{res.helper_id}`)\nScore: {res.score:.2f} | {is_hit}")

        with col_c:
            st.markdown("#### S-BERT English")
            r = get_retriever("sbert_en", profiles)
            results = r.search(selected_query_obj.text, top_k=3, eligible_helper_ids=eligible_ids)
            for res in results:
                is_hit = "🎯 Relevant" if res.helper_id in selected_query_obj.relevant_helper_ids else "❌ Non-relevant"
                st.success(f"**{res.profile.name}** (`{res.helper_id}`)\nScore: {res.score:.2f} | {is_hit}")

        with col_d:
            st.markdown("#### S-BERT Multilingual")
            r = get_retriever("sbert_multi", profiles)
            results = r.search(selected_query_obj.text, top_k=3, eligible_helper_ids=eligible_ids)
            for res in results:
                is_hit = "🎯 Relevant" if res.helper_id in selected_query_obj.relevant_helper_ids else "❌ Non-relevant"
                st.success(f"**{res.profile.name}** (`{res.helper_id}`)\nScore: {res.score:.2f} | {is_hit}")
