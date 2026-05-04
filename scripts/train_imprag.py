import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from imprag import ImpRAGConfig, ImpRAGModel, joint_loss, kl_retrieval_loss, multi_label_nce_loss
from imprag.data import ImpRAGCollator, JsonlImpRAGDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Base ImpRAG training loop.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--config-out", type=str, default="outputs/imprag_config.json")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--retrieval-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-query-length", type=int, default=512)
    parser.add_argument("--max-passage-length", type=int, default=192)
    return parser.parse_args()


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_candidate_log_likelihoods(model: ImpRAGModel, batch) -> torch.Tensor:
    scores = []
    candidate_ids = batch["candidate_input_ids"][0]
    candidate_mask = batch["candidate_attention_mask"][0]
    for idx in range(candidate_ids.size(0)):
        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=batch["labels"].to(model.device),
            passage_input_ids=candidate_ids[idx : idx + 1].to(model.device),
            passage_attention_mask=candidate_mask[idx : idx + 1].to(model.device),
        )
        scores.append((-outputs["loss"]).detach())
    return torch.stack(scores, dim=0).unsqueeze(0)


def main():
    args = parse_args()
    if args.config:
        config = ImpRAGConfig.from_json(args.config)
        config.total_epochs = args.epochs
        config.warmup_epochs = args.warmup_epochs
        config.learning_rate = args.learning_rate
        config.retrieval_loss_weight = args.retrieval_loss_weight
        config.max_query_length = args.max_query_length
        config.max_passage_length = args.max_passage_length
        config.output_dir = str(Path(args.config_out).parent)
    else:
        config = ImpRAGConfig(
            model_name_or_path=args.model_name,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            learning_rate=args.learning_rate,
            retrieval_loss_weight=args.retrieval_loss_weight,
            max_query_length=args.max_query_length,
            max_passage_length=args.max_passage_length,
            output_dir=str(Path(args.config_out).parent),
        )

    rows = load_rows(args.train_jsonl)
    model = ImpRAGModel(config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    collator = ImpRAGCollator(model.tokenizer, config.max_query_length, config.max_passage_length)
    dataloader = DataLoader(JsonlImpRAGDataset(rows), batch_size=1, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(dataloader) * config.total_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=max(1, int(total_steps * config.warmup_ratio)),
        num_training_steps=total_steps,
    )

    for epoch in range(config.total_epochs):
        model.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.total_epochs}")
        for batch in progress:
            optimizer.zero_grad()
            batch = {
                key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                passage_input_ids=batch["candidate_input_ids"][0],
                passage_attention_mask=batch["candidate_attention_mask"][0],
            )
            retrieval_scores = model.score_candidates(
                query_input_ids=batch["input_ids"],
                query_attention_mask=batch["attention_mask"],
                candidate_input_ids=batch["candidate_input_ids"],
                candidate_attention_mask=batch["candidate_attention_mask"],
            )

            if epoch < config.warmup_epochs:
                retrieval_loss = multi_label_nce_loss(
                    retrieval_scores,
                    positive_mask=batch["positive_mask"],
                    negative_mask=batch["negative_mask"],
                )
            else:
                lm_log_likelihoods = compute_candidate_log_likelihoods(model, batch)
                retrieval_loss = kl_retrieval_loss(
                    retrieval_scores=retrieval_scores,
                    lm_log_likelihoods=lm_log_likelihoods,
                    retrieval_temperature=config.retrieval_temperature,
                    target_temperature=config.target_temperature,
                )

            loss = joint_loss(outputs["loss"], retrieval_loss, config.retrieval_loss_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "gen": f"{outputs['loss'].item():.4f}",
                    "ret": f"{retrieval_loss.item():.4f}",
                }
            )

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save_json(args.config_out)
    model.lm.save_pretrained(config.output_dir)
    model.tokenizer.save_pretrained(config.output_dir)
    print(json.dumps({"config": args.config_out, "model_dir": config.output_dir}, indent=2))


if __name__ == "__main__":
    main()
