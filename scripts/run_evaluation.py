"""Execution script to run benchmarks and generate the official evaluation report."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_profiles_from_csv,
    load_queries_from_json,
    load_labels_from_json,
)
from src.evaluate import run_model_benchmark, generate_markdown_report


def main():
    print("=" * 70)
    print("SkillShare Match: Executing Official Evaluation Benchmark")
    print("=" * 70)

    # 1. Load data
    profiles = load_profiles_from_csv("data/profiles.csv")
    queries = load_queries_from_json("data/queries.json")
    labels = load_labels_from_json("data/labels.json")

    test_queries = [q for q in queries if q.split == "test"]
    print(f"Loaded {len(profiles)} profiles ({sum(1 for p in profiles if p.active)} active).")
    print(f"Evaluating across {len(test_queries)} held-out TEST queries.")

    # 2. Benchmark with hard constraint filters (Primary)
    print("\n[1/3] Benchmarking methods with Hard Constraint Filters (TF-IDF, BM25, S-BERT)...")
    eval_results = run_model_benchmark(
        profiles=profiles,
        queries=test_queries,
        labels=labels,
        methods=["tfidf", "bm25", "sbert_en", "sbert_multi"],
        top_k=3,
        min_threshold=0.15,
        apply_hard_constraints=True,
    )

    # 3. Ablation: Benchmark without constraint filters
    print("\n[2/3] Running Ablation: Evaluating models WITHOUT Constraint Filters...")
    ablation_no_filters = run_model_benchmark(
        profiles=profiles,
        queries=test_queries,
        labels=labels,
        methods=["sbert_multi"],
        top_k=3,
        min_threshold=0.15,
        apply_hard_constraints=False,
    )

    ablation_results = {
        "with_filters": eval_results,
        "without_filters": ablation_no_filters,
    }

    # 4. Generate report
    print("\n[3/3] Generating official report in reports/evaluation_report.md...")
    report_text = generate_markdown_report(
        eval_results=eval_results,
        ablation_results=ablation_results,
        split_name="Test Split (Held-out 25 Queries)",
        output_path="reports/evaluation_report.md",
    )

    print("\nSuccessfully generated reports/evaluation_report.md!")
    print("=" * 70)


if __name__ == "__main__":
    main()
