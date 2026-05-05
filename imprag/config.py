from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ImpRAGConfig:
    model_name_or_path: str = "meta-llama/Llama-3.2-3B"
    retriever_bottom_layer: int = 7
    reader_top_cross_attn_layer: int = 19
    top_k: int = 10
    max_query_length: int = 512
    max_passage_length: int = 192
    retrieval_loss_weight: float = 0.1
    warmup_epochs: int = 3
    total_epochs: int = 10
    train_batch_size: int = 1
    eval_batch_size: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    nce_top_positives: int = 5
    hard_negative_rank_low: int = 10
    hard_negative_rank_high: int = 50
    hard_negative_count: int = 8
    retrieval_temperature: float = 1.0
    target_temperature: float = 1.0
    disable_top_cross_attention: bool = True
    passage_encoding_strategy: str = "independent"
    experiment_name: str = "imprag-baseline"
    use_attention_queries_for_query_embedding: bool = True
    text_column: str = "query"
    answer_column: str = "answer"
    candidate_passages_column: str = "candidate_passages"
    output_dir: str = "outputs"
    device: Optional[str] = None
    torch_dtype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str) -> "ImpRAGConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
