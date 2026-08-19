import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.cache_utils import DynamicCache
from imprag.model import ImpRAGModel, extract_kv_for_layer, safe_update_dynamic_cache, pad_cache_to_num_layers

class AdaptiveGQAPooling(nn.Module):
    """
    Dimension 4: Adaptive GQA Attention Head Pooling.
    Replaces static uniform head averaging with query-conditioned dynamic head weighting:
    alpha_{h, i}(q) = Softmax(W_h * q_{h, i} + b_h)
    E_q,h = sum_{i=1}^g alpha_{h, i}(q) * q_{h, i}
    """
    def __init__(self, num_kv_heads=8, g=4, head_dim=128):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.g = g
        self.head_dim = head_dim
        
        # Scoring projection per head: [num_kv_heads, g, head_dim]
        # Initialized to near-zero so initial weights are almost uniform Softmax
        self.head_scores = nn.Parameter(torch.zeros(num_kv_heads, g, head_dim))
        self.scale = 1.0 / (head_dim ** 0.5)

    def forward(self, q_reshaped):
        """
        q_reshaped: Tensor of shape [batch_size, num_kv_heads, g, head_dim]
        Returns:
            E_q: Pooled query embedding of shape [batch_size, num_kv_heads * head_dim]
            attn_weights: Attention weights of shape [batch_size, num_kv_heads, g]
        """
        batch_size = q_reshaped.shape[0]
        
        # Compute dot product scores with learned parameter: [batch_size, num_kv_heads, g]
        scores = (q_reshaped * self.head_scores.unsqueeze(0)).sum(dim=-1) * self.scale
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_kv_heads, g]
        
        # Weighted sum of query heads: [batch_size, num_kv_heads, head_dim]
        q_weighted = (q_reshaped * attn_weights.unsqueeze(-1)).sum(dim=2)
        
        # Flatten across key heads: [batch_size, num_kv_heads * head_dim]
        E_q = q_weighted.view(batch_size, self.num_kv_heads * self.head_dim)
        return E_q, attn_weights

class DynamicRetrievalGate(nn.Module):
    """
    Dimension 1: Dynamic Retrieval Decision (When to Retrieve).
    Decides whether a query is parametric (can be answered from internal knowledge)
    or requires external retrieval from FAISS.
    """
    def __init__(self, input_dim=1024, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, E_q):
        """
        E_q: [batch_size, input_dim]
        Returns:
            prob_retrieve: [batch_size, 1] probability of needing retrieval
            should_retrieve: [batch_size] boolean tensor
        """
        prob = self.gate_net(E_q).squeeze(-1)  # [batch_size]
        should_retrieve = prob >= self.threshold
        return prob, should_retrieve

class AdaptiveKAllocator(nn.Module):
    """
    Dimension 2: Dynamic k / Adaptive Passage Allocation (How Much to Retrieve).
    Adaptively selects k in {1, 2, 5, 10} based on retriever similarity score distribution
    entropy and top-1 vs top-2 score margin.
    """
    def __init__(self, candidate_k_values=[1, 2, 5, 10], tau=0.1):
        super().__init__()
        self.candidate_k_values = candidate_k_values
        self.tau = tau

    def compute_k(self, scores_np):
        """
        scores_np: numpy array of top similarity scores [num_candidates] (e.g. 10 scores)
        Returns:
            optimal_k: integer in {1, 2, 5, 10}
            metadata: dict with entropy, top-margin, and normalized distribution
        """
        if len(scores_np) < 2:
            return 1, {"entropy": 0.0, "margin": 1.0}
            
        scores = scores_np[:10]
        # Softmax over top scores with temperature and safe clipping
        scaled_diff = (scores - np.max(scores)) / max(self.tau, 1e-4)
        scaled_diff = np.clip(scaled_diff, -50.0, 0.0)
        exp_s = np.exp(scaled_diff)
        probs = exp_s / np.sum(exp_s)
        
        # Normalized Shannon entropy in [0, 1]
        num_items = len(probs)
        entropy = - np.sum(probs * np.log(probs + 1e-12)) / np.log(num_items)
        margin = float(scores[0] - scores[1])
        
        # Adaptive Decision Logic:
        # 1. High margin + low entropy -> simple clear answer -> k=1
        if margin > 0.15 or entropy < 0.35:
            k = 1
        # 2. Moderate entropy -> k=2
        elif entropy < 0.60:
            k = 2
        # 3. Medium-high entropy -> k=5
        elif entropy < 0.85:
            k = 5
        # 4. Very diffuse / high entropy (multi-hop / broad topic) -> k=10
        else:
            k = 10
            
        return k, {
            "entropy": float(entropy),
            "margin": float(margin),
            "top_scores": scores[:min(5, len(scores))].tolist()
        }

class AdaptiveLayerBoundaryRouter(nn.Module):
    """
    Dimension 3: Adaptive Layer Slicing & Injection Boundaries (b(q), t(q)).
    Dynamically adjusts cache depth [b(q), t(q)] based on task complexity:
    - Tier 1 (Shallow / Fast context-grounding, e.g. phrase denoising): [b=4, t=14] (for 32-layer)
    - Tier 2 (Standard fact lookup / QA): [b=7, t=20]
    - Tier 3 (Deep multi-hop reasoning / summarization): [b=7, t=26]
    """
    def __init__(self, input_dim=1024, num_layers=32):
        super().__init__()
        self.num_layers = num_layers
        
        # Define 3 discrete complexity tiers matching model capacity
        if num_layers == 32:
            self.tiers = [
                {"name": "Shallow (Fast Grounding)", "b": 4, "t": 14},
                {"name": "Standard (Fact Lookup)", "b": 7, "t": 20},
                {"name": "Deep (Multi-Hop Reasoning)", "b": 7, "t": 26}
            ]
        elif num_layers == 28:
            self.tiers = [
                {"name": "Shallow (Fast Grounding)", "b": 4, "t": 12},
                {"name": "Standard (Fact Lookup)", "b": 7, "t": 18},
                {"name": "Deep (Multi-Hop Reasoning)", "b": 7, "t": 24}
            ]
        else:
            self.tiers = [
                {"name": "Shallow", "b": max(0, int(num_layers * 0.2)), "t": max(1, int(num_layers * 0.5))},
                {"name": "Standard", "b": max(0, int(num_layers * 0.25)), "t": max(1, int(num_layers * 0.7))},
                {"name": "Deep", "b": max(0, int(num_layers * 0.25)), "t": max(1, int(num_layers * 0.85))}
            ]
            
        self.router_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, len(self.tiers))
        )

    def forward(self, E_q):
        """
        E_q: [batch_size, input_dim]
        Returns:
            tier_probs: [batch_size, 3] probability distribution over boundary tiers
            selected_tier_indices: [batch_size] optimal tier index
        """
        logits = self.router_net(E_q)
        probs = F.softmax(logits, dim=-1)
        selected_indices = torch.argmax(probs, dim=-1)
        return probs, selected_indices

from imprag.chunking import AdaptiveRelevanceFilter, AdaptiveSemanticChunker

class AdaptiveImpRAGModel(nn.Module):
    """
    Unified ADAPTIVE ImpRAG Architecture.
    Encapsulates all 4 core dimensions:
    1. Dynamic Retrieval Decision (When to Retrieve): Bypasses FAISS for parametric queries.
    2. Dynamic k Budget Allocation (How Much to Retrieve): Chooses k in {1, 2, 5, 10}.
    3. Adaptive Layer Slicing [b(q), t(q)]: Adjusts injection depth by task complexity.
    4. Adaptive GQA Attention Head Pooling: Dynamic learned head attention weighting.
    + Integrated Semantic Relevance Filtering & Deterministic Greedy Decoding.
    """
    def __init__(self, base_model, default_b=7, default_t=23, pooling_type="last_token"):
        super().__init__()
        self.base_model = base_model
        self.base_imprag = ImpRAGModel(
            base_model=base_model,
            b=default_b,
            t=default_t,
            k_passages=5,
            max_passage_len=192,
            pooling_type=pooling_type
        )
        
        num_layers = self.base_imprag._get_num_layers()
        if hasattr(base_model.config, "num_attention_heads") and hasattr(base_model.config, "num_key_value_heads"):
            num_attn_heads = base_model.config.num_attention_heads
            num_kv_heads = base_model.config.num_key_value_heads
            head_dim = base_model.config.hidden_size // num_attn_heads
            g = num_attn_heads // num_kv_heads
        else:
            num_kv_heads = 4
            g = 1
            head_dim = 32
            
        emb_dim = num_kv_heads * head_dim
        
        # Dimension 4: Adaptive GQA Pooling
        self.adaptive_pooling = AdaptiveGQAPooling(num_kv_heads=num_kv_heads, g=g, head_dim=head_dim)
        
        # Dimension 1: Dynamic Retrieval Gate
        self.retrieval_gate = DynamicRetrievalGate(input_dim=emb_dim, threshold=0.45)
        
        # Dimension 2: Dynamic k Allocator
        self.k_allocator = AdaptiveKAllocator()
        
        # Dimension 3: Adaptive Layer Slicing Router
        self.layer_router = AdaptiveLayerBoundaryRouter(input_dim=emb_dim, num_layers=num_layers)
        
        # Advanced Relevance Filter & Re-Ranker
        self.relevance_filter = AdaptiveRelevanceFilter(lexical_weight=0.35, min_relevance_threshold=0.15)
        self.chunker = AdaptiveSemanticChunker(target_chunk_size=150, overlap_size=40)

    def get_adaptive_query_embedding(self, query_ids, attention_mask=None):
        """
        Extracts query activations at layer b and applies Adaptive GQA Head Attention pooling.
        Returns:
            E_q: [batch_size, embedding_dim]
            attn_weights: [batch_size, num_kv_heads, g] head weights
        """
        self.base_imprag.clear_captured_states()
        layers = self.base_imprag._get_layers_list()
        orig_layers = list(layers)
        
        if self.base_imprag.model_type == "llama":
            self.base_model.model.layers = nn.ModuleList(orig_layers[:self.base_imprag.b + 1])
        elif self.base_imprag.model_type == "gpt2":
            self.base_model.transformer.h = nn.ModuleList(orig_layers[:self.base_imprag.b + 1])
            
        try:
            self.base_model(
                input_ids=query_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False
            )
        finally:
            if self.base_imprag.model_type == "llama":
                self.base_model.model.layers = nn.ModuleList(orig_layers)
            elif self.base_imprag.model_type == "gpt2":
                self.base_model.transformer.h = nn.ModuleList(orig_layers)
                
        captured_q = self.base_imprag.captured_q
        self.base_imprag.clear_captured_states()
        
        batch_size = query_ids.shape[0]
        if self.base_imprag.pooling_type == "last_token":
            if attention_mask is not None:
                last_indices = (attention_mask.sum(dim=-1) - 1).clamp(min=0)
            else:
                last_indices = torch.full((batch_size,), query_ids.shape[1] - 1, dtype=torch.long, device=query_ids.device)
            q_pooled = torch.stack([captured_q[i, last_indices[i]] for i in range(batch_size)])
        else:
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                q_pooled = (captured_q * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                q_pooled = captured_q.mean(dim=1)
                
        if self.base_imprag.model_type == "llama":
            num_attn_heads = self.base_model.config.num_attention_heads
            num_kv_heads = self.base_model.config.num_key_value_heads
            head_dim = self.base_model.config.hidden_size // num_attn_heads
            g = num_attn_heads // num_kv_heads
            q_reshaped = q_pooled.view(batch_size, num_kv_heads, g, head_dim)
            E_q, head_weights = self.adaptive_pooling(q_reshaped)
        else:
            E_q = q_pooled.view(batch_size, -1)
            head_weights = None
            
        return E_q, head_weights

    def encode_adaptive_passages(self, passage_ids, b_layer, t_layer, k_passages, attention_mask=None):
        """
        Encodes passages dynamically into custom layer boundaries [b_layer .. t_layer].
        """
        orig_b, orig_t = self.base_imprag.b, self.base_imprag.t
        self.base_imprag.b = b_layer
        self.base_imprag.t = t_layer
        try:
            cache = self.base_imprag.encode_passages(
                passage_ids=passage_ids,
                attention_mask=attention_mask,
                k_passages=k_passages,
                encoding_mode="concatenated"
            )
            return cache
        finally:
            self.base_imprag.b = orig_b
            self.base_imprag.t = orig_t

    def adaptive_generate(self, query_ids, query_text, faiss_index, passages, tokenizer, max_new_tokens=40, force_retrieve=None, temperature=0.0):
        """
        Full Adaptive ImpRAG Inference Pipeline:
        1. Adaptive GQA pooling -> E_q and head weights.
        2. Dynamic Retrieval Decision -> Check if retrieval is necessary.
        3. If bypassed: direct parametric generation (k=0) with deterministic greedy decoding.
        4. If retrieved: FAISS search -> Hybrid Relevance Filtering & Re-Ranking -> Dynamic k allocation -> Dynamic boundary routing [b(q), t(q)].
        5. Full Attention Concatenated Passage Encoding into DynamicCache.
        6. Generation + Comprehensive Telemetry.
        """
        self.eval()
        device = query_ids.device
        
        # Determine EOS token IDs
        eos_ids = {tokenizer.eos_token_id}
        if self.base_imprag.model_type == "llama":
            eos_ids.update([128001, 128009])
            
        # Format reader prompt if using Llama-3-Instruct
        if self.base_imprag.model_type == "llama" and "<|start_header_id|>" not in query_text:
            instruct_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question directly and concisely based on the knowledge provided:\n{query_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            reader_query_ids = tokenizer([instruct_prompt], return_tensors="pt").input_ids.to(device)
        else:
            reader_query_ids = query_ids
        
        with torch.no_grad():
            # 1. Dynamic Query Embedding with Adaptive GQA Head Pooling
            E_q, head_weights = self.get_adaptive_query_embedding(query_ids)
            E_q_norm = F.normalize(E_q, p=2, dim=-1)
            E_q_np = E_q_norm.float().cpu().numpy()
            
            # 2. Dynamic Retrieval Decision Gate
            prob_ret, should_ret = self.retrieval_gate(E_q_norm)
            needs_retrieval = should_ret[0].item() if force_retrieve is None else force_retrieve
            
            telemetry = {
                "query": query_text,
                "retrieval_prob": float(prob_ret[0].item()),
                "retrieval_decision": "RETRIEVED" if needs_retrieval else "PARAMETRIC_BYPASS",
                "head_attention_weights": head_weights[0].cpu().numpy().tolist() if head_weights is not None else None,
                "k_allocated": 0,
                "layer_boundaries": None,
                "entropy": None,
                "margin": None,
                "retrieved_passages": [],
                "compute_saved": "0%"
            }
            
            # CASE A: Parametric Query -> Bypass Retrieval Entirely!
            if not needs_retrieval:
                telemetry["compute_saved"] = "~85% (FAISS + Passage KV Encoding Bypassed)"
                gen_tokens = self.base_imprag.generate(
                    reader_query_ids, 
                    custom_past_key_values=None, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_ids=eos_ids
                )
                gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
                return gen_text, telemetry
                
            # CASE B: Low-Confidence / Non-Parametric Query -> Trigger Retrieval
            # Search FAISS for top candidates
            distances, indices = faiss_index.search(E_q_np, k=15)
            raw_scores = distances[0]
            raw_indices = indices[0]
            
            valid_passages = [passages[idx] for idx in raw_indices if idx != -1 and idx < len(passages)]
            valid_scores = [raw_scores[i] for i, idx in enumerate(raw_indices) if idx != -1 and idx < len(passages)]
            
            # Apply Hybrid Semantic & Lexical Relevance Filter & Re-Ranker
            filtered_passages, filtered_scores = self.relevance_filter.filter_and_rerank(
                query_text=query_text,
                candidate_passages=valid_passages,
                dense_scores=valid_scores
            )
            
            if not filtered_passages:
                filtered_passages = ["No relevant context found."]
                filtered_scores = np.array([0.0])
            else:
                filtered_scores = np.array(filtered_scores)
            
            # 3. Dynamic k Allocation based on filtered relevance distribution
            k_opt, k_meta = self.k_allocator.compute_k(filtered_scores)
            selected_passages = filtered_passages[:k_opt]
            
            telemetry["k_allocated"] = len(selected_passages)
            telemetry["entropy"] = k_meta["entropy"]
            telemetry["margin"] = k_meta["margin"]
            telemetry["retrieved_passages"] = selected_passages
            
            # 4. Adaptive Layer Slicing [b(q), t(q)]
            tier_probs, tier_idx = self.layer_router(E_q_norm)
            selected_tier = self.layer_router.tiers[tier_idx[0].item()]
            b_q, t_q = selected_tier["b"], selected_tier["t"]
            telemetry["layer_boundaries"] = {"b": b_q, "t": t_q, "tier_name": selected_tier["name"]}
            
            # 5. Full Attention Concatenated Passage Encoding with Expanded Capacity
            passage_enc = tokenizer(
                selected_passages,
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt"
            ).input_ids.to(device).unsqueeze(0)
            
            custom_cache = self.encode_adaptive_passages(
                passage_ids=passage_enc,
                b_layer=b_q,
                t_layer=t_q,
                k_passages=len(selected_passages)
            )
            
            # 6. Autoregressive Deterministic Generation with Prepended Passage Cache
            orig_b = self.base_imprag.b
            self.base_imprag.b = b_q
            try:
                gen_tokens = self.base_imprag.generate(
                    query_ids=reader_query_ids,
                    custom_past_key_values=custom_cache,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_ids=eos_ids
                )
            finally:
                self.base_imprag.b = orig_b
                
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
            
            # Compute saved estimate vs static k=5
            k_baseline = 5
            saved_ratio = max(0.0, (k_baseline - len(selected_passages)) / k_baseline * 100)
            telemetry["compute_saved"] = f"{saved_ratio:.1f}% vs static k=5"
            
            return gen_text, telemetry
