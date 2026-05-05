import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from imprag import FaissRetriever, ImpRAGConfig, ImpRAGModel
from imprag.retrieval import load_passages, sample_passages


def parse_args():
    parser = argparse.ArgumentParser(description="Build a FAISS index for ImpRAG passages.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--passages", type=str, required=True, help="JSONL with {id, text}")
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ImpRAGConfig.from_json(args.config) if args.config else ImpRAGConfig(model_name_or_path=args.model_name)
    model = ImpRAGModel(config)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    passages = sample_passages(load_passages(args.passages), sample_size=args.sample_size, seed=args.seed)
    embeddings = []
    ids = []
    texts = []

    for start in tqdm(range(0, len(passages), args.batch_size), desc="Encoding passages"):
        batch = passages[start : start + args.batch_size]
        tokens = model.tokenizer(
            [item["text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_passage_length,
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        with torch.no_grad():
            batch_embeddings = model.encode_passages_for_retrieval(tokens["input_ids"], tokens["attention_mask"])
        embeddings.append(batch_embeddings.cpu().numpy())
        ids.extend([item["id"] for item in batch])
        texts.extend([item["text"] for item in batch])

    embeddings = np.concatenate(embeddings, axis=0).astype("float32")
    retriever = FaissRetriever(dim=embeddings.shape[1])
    retriever.add(embeddings, ids=ids, texts=texts)
    Path(args.index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)
    retriever.save(args.index_path, args.metadata_path)
    print(json.dumps({"index_path": args.index_path, "metadata_path": args.metadata_path, "passages": len(passages)}, indent=2))


if __name__ == "__main__":
    main()
