import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline vs ImpRAG+ evaluation outputs.")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    summary = {
        "baseline_run": baseline["run_name"],
        "candidate_run": candidate["run_name"],
        "baseline_exact_match": baseline["exact_match"],
        "candidate_exact_match": candidate["exact_match"],
        "delta_exact_match": candidate["exact_match"] - baseline["exact_match"],
        "baseline_cleaned_exact_match": baseline.get("cleaned_exact_match", baseline["exact_match"]),
        "candidate_cleaned_exact_match": candidate.get("cleaned_exact_match", candidate["exact_match"]),
        "delta_cleaned_exact_match": candidate.get("cleaned_exact_match", candidate["exact_match"]) - baseline.get("cleaned_exact_match", baseline["exact_match"]),
        "baseline_retrieval_recall_at_k": baseline["retrieval_recall_at_k"],
        "candidate_retrieval_recall_at_k": candidate["retrieval_recall_at_k"],
        "delta_retrieval_recall_at_k": candidate["retrieval_recall_at_k"] - baseline["retrieval_recall_at_k"],
        "examples": candidate["examples"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
