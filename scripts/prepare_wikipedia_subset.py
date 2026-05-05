import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a Wikipedia-style JSONL corpus for ImpRAG/ImpRAG+.")
    parser.add_argument("--dataset-name", type=str, default="kilt_wikipedia")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="full")
    parser.add_argument("--fallback-dataset-name", type=str, default="wikimedia/wikipedia")
    parser.add_argument("--fallback-dataset-config", type=str, default="20231101.en")
    parser.add_argument("--fallback-split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--id-column", type=str, default=None)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--title-column", type=str, default=None)
    parser.add_argument("--max-characters", type=int, default=1200)
    parser.add_argument("--append-title", action="store_true")
    return parser.parse_args()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces = [_normalize_text(item) for item in value]
        return "\n".join(piece for piece in pieces if piece)
    if isinstance(value, dict):
        if "paragraph" in value:
            return _normalize_text(value["paragraph"])
        if "text" in value:
            return _normalize_text(value["text"])
        if "value" in value:
            return _normalize_text(value["value"])
    return str(value).strip()


def _candidate_value(row: dict, *names: str) -> Optional[Any]:
    for name in names:
        if name and name in row and row[name] is not None:
            return row[name]
    return None


def row_to_passage(row: dict, index: int, args) -> Optional[dict]:
    passage_id = _candidate_value(row, args.id_column, "id", "wikipedia_id", "wiki_id", "docid")
    title = _normalize_text(_candidate_value(row, args.title_column, "title"))
    text_value = _candidate_value(row, args.text_column, "text", "paragraphs", "contents")
    text = _normalize_text(text_value)
    if args.append_title and title:
        text = f"{title}\n{text}" if text else title
    text = text[: args.max_characters].strip()
    if not text:
        return None
    return {
        "id": str(passage_id if passage_id is not None else index),
        "title": title,
        "text": text,
    }


def choose_rows(rows: Iterable[dict], sample_size: Optional[int], seed: int) -> list[dict]:
    rows = list(rows)
    if sample_size is None or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    indices = sorted(indices[:sample_size])
    return [rows[idx] for idx in indices]


def choose_rows_streaming(rows: Iterable[dict], sample_size: Optional[int]) -> list[dict]:
    selected = []
    for row in rows:
        selected.append(row)
        if sample_size is not None and len(selected) >= sample_size:
            break
    return selected


def load_rows_with_fallback(args):
    try:
        if args.streaming:
            dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split, streaming=True)
            if args.sample_size is not None:
                dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
            return choose_rows_streaming(dataset, args.sample_size), {
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "fallback_used": False,
            }

        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        return choose_rows(dataset, args.sample_size, args.seed), {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "fallback_used": False,
        }
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        fallback_name = args.fallback_dataset_name
        fallback_config = args.fallback_dataset_config
        fallback_split = args.fallback_split
        print(
            json.dumps(
                {
                    "warning": "Primary dataset uses a deprecated dataset script. Falling back to a supported Wikipedia source.",
                    "primary_dataset": args.dataset_name,
                    "fallback_dataset": fallback_name,
                    "fallback_config": fallback_config,
                    "fallback_split": fallback_split,
                },
                indent=2,
            )
        )
        dataset = load_dataset(fallback_name, fallback_config, split=fallback_split, streaming=args.streaming)
        if args.streaming:
            if args.sample_size is not None:
                dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
            rows = choose_rows_streaming(dataset, args.sample_size)
        else:
            rows = choose_rows(dataset, args.sample_size, args.seed)
        return rows, {
            "dataset_name": fallback_name,
            "dataset_config": fallback_config,
            "split": fallback_split,
            "fallback_used": True,
        }


def main():
    args = parse_args()
    selected_rows, metadata = load_rows_with_fallback(args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(selected_rows):
            passage = row_to_passage(row, idx, args)
            if passage is None:
                continue
            handle.write(json.dumps(passage, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({"output_path": str(output_path), "rows_written": written, **metadata}, indent=2))


if __name__ == "__main__":
    main()
