import argparse
import json

import torch

from imprag import FaissRetriever, ImpRAGConfig, ImpRAGModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-retrieval ImpRAG inference.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2)
    parser.add_argument("--disable-repeat-stop", action="store_true")
    parser.add_argument("--disable-answer-cleaning", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ImpRAGConfig.from_json(args.config) if args.config else ImpRAGConfig(model_name_or_path=args.model_name, top_k=args.top_k)
    if args.model_path:
        config.model_name_or_path = args.model_path
    model = ImpRAGModel(config)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    retriever = FaissRetriever.load(args.index_path, args.metadata_path)

    result = model.generate_with_retrieval(
        prompt=args.query,
        retriever=retriever,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        stop_on_repeat=not args.disable_repeat_stop,
        clean_answer=not args.disable_answer_cleaning,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
