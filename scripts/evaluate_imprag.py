import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from imprag import FaissRetriever, ImpRAGConfig, ImpRAGModel, exact_match, mean, retrieval_recall_at_k


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ImpRAG or ImpRAG+ on a JSONL eval set.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--eval-jsonl", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="imprag-run")
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_list(row: dict) -> list[str]:
    if "answers" in row:
        return [str(item) for item in row["answers"]]
    return [str(row["answer"])]


def main():
    args = parse_args()
    config = ImpRAGConfig.from_json(args.config)
    if args.model_path:
        config.model_name_or_path = args.model_path
    model = ImpRAGModel(config)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    retriever = FaissRetriever.load(args.index_path, args.metadata_path)
    rows = load_rows(args.eval_jsonl)

    raw_em_scores = []
    cleaned_em_scores = []
    recall_scores = []
    predictions = []

    for row in tqdm(rows, desc=f"Evaluating {args.run_name}"):
        result = model.generate_with_retrieval(
            prompt=row["query"],
            retriever=retriever,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k or config.top_k,
        )
        answers = answer_list(row)
        retrieved_texts = [item["text"] for item in result["retrieved_passages"]]
        raw_em = max(exact_match(result["answer_text"], answer) for answer in answers)
        cleaned_em = max(exact_match(result["cleaned_answer_text"], answer) for answer in answers)
        recall = retrieval_recall_at_k(answers, retrieved_texts)
        raw_em_scores.append(raw_em)
        cleaned_em_scores.append(cleaned_em)
        recall_scores.append(recall)
        predictions.append(
            {
                "query": row["query"],
                "answers": answers,
                "prediction": result["answer_text"],
                "cleaned_prediction": result["cleaned_answer_text"],
                "full_text": result["text"],
                "retrieved_passages": result["retrieved_passages"],
                "exact_match": raw_em,
                "cleaned_exact_match": cleaned_em,
                "retrieval_recall_at_k": recall,
            }
        )

    payload = {
        "run_name": args.run_name,
        "examples": len(rows),
        "exact_match": mean(raw_em_scores),
        "cleaned_exact_match": mean(cleaned_em_scores),
        "retrieval_recall_at_k": mean(recall_scores),
        "predictions": predictions,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
