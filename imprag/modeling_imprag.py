from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ImpRAGConfig


@dataclass
class QueryEncoding:
    hidden_states: torch.Tensor
    query_embedding: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


@dataclass
class PassageCache:
    hidden_by_layer: Dict[int, torch.Tensor]
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    count: int


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_heads * n_rep, seq_len, head_dim)


def _attn_num_heads(attn: nn.Module) -> int:
    return getattr(attn, "num_heads", getattr(attn.config, "num_attention_heads"))


def _attn_num_kv_heads(attn: nn.Module) -> int:
    return getattr(attn, "num_key_value_heads", getattr(attn.config, "num_key_value_heads"))


class ImpRAGModel(nn.Module):
    def __init__(self, config: ImpRAGConfig):
        super().__init__()
        self.config = config
        dtype = getattr(torch, config.torch_dtype) if config.torch_dtype else None
        self.lm = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.decoder = self.lm.model
        self.layers = self.decoder.layers
        self.hidden_size = self.lm.config.hidden_size
        self.num_layers = len(self.layers)
        self.b = config.retriever_bottom_layer
        self.t = config.reader_top_cross_attn_layer
        if not (0 <= self.b < self.num_layers):
            raise ValueError(f"retriever_bottom_layer must be in [0, {self.num_layers - 1}]")
        if not (self.b <= self.t < self.num_layers):
            raise ValueError(f"reader_top_cross_attn_layer must be in [{self.b}, {self.num_layers - 1}]")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained_config(cls, path: str) -> "ImpRAGModel":
        return cls(ImpRAGConfig.from_json(path))

    def save_config(self, path: str) -> None:
        self.config.save_json(path)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.embed_tokens(input_ids)

    def _position_ids(self, attention_mask: torch.Tensor, shift: int = 0) -> torch.Tensor:
        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids = position_ids.clamp_min(0)
        return position_ids + shift

    def _run_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        cache_position = torch.arange(hidden_states.size(1), device=hidden_states.device)
        position_embeddings = self.decoder.rotary_emb(hidden_states, position_ids)
        layer_attention_mask = self.decoder.create_extended_attention_mask_for_decoder(
            input_shape=(hidden_states.size(0), hidden_states.size(1)),
            attention_mask=attention_mask,
            device=hidden_states.device,
        )
        outputs = layer(
            hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs if isinstance(outputs, torch.Tensor) else outputs.hidden_states

    def _project_last_token(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        use_query_projection: bool,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        proj = layer.self_attn.q_proj if use_query_projection else layer.self_attn.k_proj
        proj_states = proj(hidden_states)

        if use_query_projection:
            num_heads = _attn_num_heads(layer.self_attn)
            num_kv_heads = _attn_num_kv_heads(layer.self_attn)
            group_size = num_heads // num_kv_heads
            proj_states = proj_states.view(hidden_states.size(0), hidden_states.size(1), num_heads, -1)
            proj_states = proj_states.view(hidden_states.size(0), hidden_states.size(1), num_kv_heads, group_size, -1)
            proj_states = proj_states.mean(dim=3)
        else:
            num_kv_heads = _attn_num_kv_heads(layer.self_attn)
            proj_states = proj_states.view(hidden_states.size(0), hidden_states.size(1), num_kv_heads, -1)

        lengths = attention_mask.sum(dim=-1) - 1
        pooled = []
        for batch_idx, token_idx in enumerate(lengths.tolist()):
            pooled.append(proj_states[batch_idx, token_idx].reshape(-1))
        pooled = torch.stack(pooled, dim=0)
        return F.normalize(pooled, dim=-1)

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> QueryEncoding:
        hidden_states = self._embed(input_ids)
        position_ids = self._position_ids(attention_mask)
        for layer_idx in range(self.b + 1):
            hidden_states = self._run_layer(self.layers[layer_idx], hidden_states, attention_mask, position_ids)

        query_embedding = self._project_last_token(
            layer_idx=self.b,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_query_projection=self.config.use_attention_queries_for_query_embedding,
        )
        return QueryEncoding(
            hidden_states=hidden_states,
            query_embedding=query_embedding,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def encode_passages_for_retrieval(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self._embed(input_ids)
        position_ids = self._position_ids(attention_mask)
        for layer_idx in range(self.b + 1):
            hidden_states = self._run_layer(self.layers[layer_idx], hidden_states, attention_mask, position_ids)
        return self._project_last_token(
            layer_idx=self.b,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_query_projection=False,
        )

    def encode_passages_for_cross_attention(
        self,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
        strategy: Optional[str] = None,
    ) -> PassageCache:
        strategy = strategy or self.config.passage_encoding_strategy
        if passage_input_ids.dim() != 2:
            raise ValueError("Expected passage_input_ids with shape [num_passages, seq_len]")
        if strategy != "independent":
            raise NotImplementedError(
                "This base repo implements independent passage encoding first. "
                "The paper also studies segmented and full-attention concatenation."
            )

        hidden_states = self._embed(passage_input_ids)
        position_ids = self._position_ids(passage_attention_mask)
        hidden_by_layer: Dict[int, torch.Tensor] = {}
        for layer_idx in range(self.t + 1):
            hidden_states = self._run_layer(self.layers[layer_idx], hidden_states, passage_attention_mask, position_ids)
            if self.b <= layer_idx <= self.t:
                hidden_by_layer[layer_idx] = hidden_states.detach()

        flat_attention_mask = passage_attention_mask.reshape(1, -1)
        flat_position_ids = position_ids.reshape(1, -1)
        flattened_hidden_by_layer = {
            layer_idx: layer_hidden.reshape(1, -1, layer_hidden.size(-1))
            for layer_idx, layer_hidden in hidden_by_layer.items()
        }
        return PassageCache(
            hidden_by_layer=flattened_hidden_by_layer,
            attention_mask=flat_attention_mask,
            position_ids=flat_position_ids,
            count=passage_input_ids.size(0),
        )

    def compute_retrieval_scores(self, query_embedding: torch.Tensor, passage_embeddings: torch.Tensor) -> torch.Tensor:
        if passage_embeddings.dim() == 2:
            passage_embeddings = passage_embeddings.unsqueeze(0)
        return torch.einsum("bd,bnd->bn", query_embedding, passage_embeddings)

    def _cross_attention(
        self,
        layer_idx: int,
        query_hidden_states: torch.Tensor,
        passage_hidden_states: torch.Tensor,
        passage_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        attn = layer.self_attn

        batch_size, query_length, _ = query_hidden_states.shape
        _, passage_length, _ = passage_hidden_states.shape

        num_heads = _attn_num_heads(attn)
        num_kv_heads = _attn_num_kv_heads(attn)
        query_states = attn.q_proj(query_hidden_states).view(batch_size, query_length, num_heads, attn.head_dim).transpose(1, 2)
        key_states = attn.k_proj(passage_hidden_states).view(batch_size, passage_length, num_kv_heads, attn.head_dim).transpose(1, 2)
        value_states = attn.v_proj(passage_hidden_states).view(batch_size, passage_length, num_kv_heads, attn.head_dim).transpose(1, 2)
        key_states = _repeat_kv(key_states, attn.num_key_value_groups)
        value_states = _repeat_kv(value_states, attn.num_key_value_groups)

        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / (attn.head_dim ** 0.5)
        key_mask = passage_attention_mask[:, None, None, :].to(dtype=scores.dtype)
        scores = scores.masked_fill(key_mask == 0, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probs, value_states)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        return attn.o_proj(attended)

    def decode_with_passages(
        self,
        query_hidden_states: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_cache: Optional[PassageCache],
    ) -> torch.Tensor:
        position_shift = 0
        if passage_cache is not None:
            position_shift = passage_cache.count * self.config.max_passage_length
        query_position_ids = self._position_ids(query_attention_mask, shift=position_shift)

        hidden_states = query_hidden_states
        for layer_idx in range(self.b + 1, self.t + 1):
            hidden_states = self._run_layer(self.layers[layer_idx], hidden_states, query_attention_mask, query_position_ids)
            if passage_cache is not None:
                hidden_states = hidden_states + self._cross_attention(
                    layer_idx=layer_idx,
                    query_hidden_states=hidden_states,
                    passage_hidden_states=passage_cache.hidden_by_layer[layer_idx],
                    passage_attention_mask=passage_cache.attention_mask,
                )

        for layer_idx in range(self.t + 1, self.num_layers):
            hidden_states = self._run_layer(self.layers[layer_idx], hidden_states, query_attention_mask, query_position_ids)
            if passage_cache is not None and not self.config.disable_top_cross_attention:
                hidden_states = hidden_states + self._cross_attention(
                    layer_idx=layer_idx,
                    query_hidden_states=hidden_states,
                    passage_hidden_states=passage_cache.hidden_by_layer.get(self.t, next(iter(passage_cache.hidden_by_layer.values()))),
                    passage_attention_mask=passage_cache.attention_mask,
                )

        hidden_states = self.decoder.norm(hidden_states)
        return self.lm.lm_head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        passage_input_ids: Optional[torch.Tensor] = None,
        passage_attention_mask: Optional[torch.Tensor] = None,
        passage_cache: Optional[PassageCache] = None,
    ) -> Dict[str, torch.Tensor]:
        query = self.encode_query(input_ids=input_ids, attention_mask=attention_mask)
        if passage_cache is None and passage_input_ids is not None and passage_attention_mask is not None:
            passage_cache = self.encode_passages_for_cross_attention(
                passage_input_ids=passage_input_ids,
                passage_attention_mask=passage_attention_mask,
            )
        logits = self.decode_with_passages(query.hidden_states, query.attention_mask, passage_cache)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {
            "loss": loss,
            "logits": logits,
            "query_embedding": query.query_embedding,
        }

    @torch.no_grad()
    def score_candidates(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_encoding = self.encode_query(query_input_ids, query_attention_mask)
        flat_candidate_ids = candidate_input_ids.squeeze(0)
        flat_candidate_mask = candidate_attention_mask.squeeze(0)
        passage_embeddings = self.encode_passages_for_retrieval(flat_candidate_ids, flat_candidate_mask)
        return self.compute_retrieval_scores(query_encoding.query_embedding, passage_embeddings.unsqueeze(0))

    @torch.no_grad()
    def generate_with_retrieval(
        self,
        prompt: str,
        retriever,
        max_new_tokens: int = 64,
        top_k: Optional[int] = None,
    ) -> Dict[str, object]:
        top_k = top_k or self.config.top_k
        query_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_query_length)
        query_tokens = {k: v.to(self.device) for k, v in query_tokens.items()}
        query_encoding = self.encode_query(query_tokens["input_ids"], query_tokens["attention_mask"])
        hits = retriever.search(query_encoding.query_embedding.detach().cpu().numpy(), top_k=top_k)[0]
        passage_texts = [item["text"] for item in hits]
        passage_tokens = self.tokenizer(
            passage_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_passage_length,
        )
        passage_tokens = {k: v.to(self.device) for k, v in passage_tokens.items()}
        passage_cache = self.encode_passages_for_cross_attention(
            passage_tokens["input_ids"],
            passage_tokens["attention_mask"],
        )

        generated = query_tokens["input_ids"]
        generated_mask = query_tokens["attention_mask"]
        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated,
                attention_mask=generated_mask,
                passage_cache=passage_cache,
            )
            next_token = outputs["logits"][:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            generated_mask = torch.cat([generated_mask, torch.ones_like(next_token)], dim=-1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return {
            "text": text,
            "retrieved_passages": hits,
        }
