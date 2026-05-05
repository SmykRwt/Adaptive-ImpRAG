import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from imprag import FaissRetriever, ImpRAGConfig, ImpRAGModel


def parse_args():
    parser = argparse.ArgumentParser(description="Build candidate-passage training data from QA examples and a retrieval index.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--qa-jsonl", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--retrieve-depth", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-hard-negatives", type=int, default=3)
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def contains_any_answer(text: str, answers: list[str]) -> bool:
    lowered = normalize(text)
    return any(normalize(answer) in lowered for answer in answers if answer)


def main():
    args = parse_args()
    config = ImpRAGConfig.from_json(args.config)
    model = ImpRAGModel(config)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    retriever = FaissRetriever.load(args.index_path, args.metadata_path)
    qa_rows = load_rows(args.qa_jsonl)

    output_rows = []
    for row in tqdm(qa_rows, desc="Building retrieval training data"):
        answers = [str(item) for item in row.get("answers", [row["answer"]])]
        result = model.generate_with_retrieval(
            prompt=row["query"],
            retriever=retriever,
            top_k=args.retrieve_depth,
            max_new_tokens=1,
            clean_answer=False,
        )
        candidates = result["retrieved_passages"][: args.max_candidates]
        positives = [idx for idx, item in enumerate(candidates) if contains_any_answer(item["text"], answers)]
        if not positives and candidates:
            positives = [0]
        hard_negatives = [idx for idx in range(len(candidates)) if idx not in positives][: args.max_hard_negatives]
        output_rows.append(
            {
                "query": row["query"],
                "answer": answers[0],
                "answers": answers,
                "candidate_passages": [{"id": item["id"], "text": item["text"]} for item in candidates],
                "positives": positives,
                "hard_negatives": hard_negatives,
                "task_type": "retrieval",
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output_path": str(output_path), "examples_written": len(output_rows)}, indent=2))


if __name__ == "__main__":
    main()
