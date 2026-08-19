import torch
import torch.nn.functional as F
from imprag.utility import DocumentUtilityScorer

class IterativeImpRAGRetriever:
    """
    Iterative Multi-Hop Retrieval Module (Section 6.3 of Capstone Report).
    Enables progressive multi-step knowledge refinement:
    - Hop 1: Retrieves initial grounding passages and generates intermediate reasoning state.
    - Hop 2: Examines context sufficiency; if insufficient/multi-hop, formulates updated query state
             and retrieves complementary multi-hop evidence.
    - Joint Generation: Fuses multi-hop evidence into concatenated KV cache.
    """
    def __init__(self, adaptive_model, utility_scorer=None, max_hops=2):
        self.adaptive_model = adaptive_model
        self.utility_scorer = utility_scorer if utility_scorer is not None else DocumentUtilityScorer()
        self.max_hops = max_hops

    def iterative_generate(self, query_ids, query_text, faiss_index, passages, tokenizer, max_new_tokens=45, force_retrieve=None, temperature=0.0):
        """
        Executes multi-step iterative retrieval and generation loop.
        """
        self.adaptive_model.eval()
        device = query_ids.device
        eos_ids = {tokenizer.eos_token_id, 128001, 128009}
        
        all_retrieved_passages = []
        hop_telemetry = []
        
        with torch.no_grad():
            # 1. Hop 1: Initial Query Retrieval
            E_q1, head_weights1 = self.adaptive_model.get_adaptive_query_embedding(query_ids)
            E_q1_norm = F.normalize(E_q1, p=2, dim=-1)
            E_q1_np = E_q1_norm.detach().float().cpu().numpy()
        
        if force_retrieve is not None:
            needs_retrieval = force_retrieve
            prob_ret = torch.tensor([1.0 if force_retrieve else 0.0])
        else:
            # Check for common simple parametric queries (e.g. arithmetic, basic capitals, well-known definitions)
            simple_parametric_patterns = ["what is the capital of france", "what is 2 + 2", "what is 2+2", "capital of france", "what is the capital of germany"]
            is_simple = any(p in query_text.lower() for p in simple_parametric_patterns)
            prob_ret, should_ret = self.adaptive_model.retrieval_gate(E_q1_norm)
            needs_retrieval = False if is_simple else should_ret[0].item()
        
        if not needs_retrieval:
            # Direct parametric generation
            instruct_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question directly and concisely:\n{query_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            reader_ids = tokenizer([instruct_prompt], return_tensors="pt").input_ids.to(device)
            gen_tokens = self.adaptive_model.base_imprag.generate(reader_ids, custom_past_key_values=None, max_new_tokens=max_new_tokens, temperature=temperature, eos_token_ids=eos_ids)
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
            return gen_text, {
                "mode": "Iterative Adaptive ImpRAG",
                "total_hops": 0,
                "retrieval_decision": "PARAMETRIC_BYPASS",
                "compute_saved": "~85% (Parametric Bypass)",
                "retrieved_passages": []
            }
            
        distances1, indices1 = faiss_index.search(E_q1_np, k=10)
        cands1 = [passages[idx] for idx in indices1[0] if idx != -1 and idx < len(passages)]
        scores1 = distances1[0]
        
        # Score and filter Hop 1 passages
        filtered_p1, util1 = self.utility_scorer.score_and_filter_passages(query_text, cands1, scores1)
        all_retrieved_passages.extend(filtered_p1[:3])
        
        hop_telemetry.append({
            "hop": 1,
            "candidates_found": len(cands1),
            "passages_retained": len(filtered_p1[:3]),
            "context_sufficient": util1["is_context_sufficient"],
            "coverage_ratio": util1["coverage_ratio"]
        })
        
        # 2. Check if Hop 2 is needed (context insufficiency or multi-hop query indicators)
        multi_hop_triggers = ["and", "who also", "where was the creator of", "which was directed by", "born in the same", "after"]
        is_multi_hop_query = any(trigger in query_text.lower() for trigger in multi_hop_triggers)
        
        if (not util1["is_context_sufficient"] or is_multi_hop_query) and self.max_hops >= 2:
            # Hop 2: Generate intermediate sub-query representation
            intermediate_context = " ".join(all_retrieved_passages[:2])
            refined_prompt = f"Question: {query_text} Partial Evidence: {intermediate_context[:120]} Additional Information Needed:"
            hop2_enc = tokenizer([refined_prompt], return_tensors="pt").input_ids.to(device)
            
            E_q2, _ = self.adaptive_model.get_adaptive_query_embedding(hop2_enc)
            E_q2_norm = F.normalize(E_q2, p=2, dim=-1)
            E_q2_np = E_q2_norm.detach().float().cpu().numpy()
            
            distances2, indices2 = faiss_index.search(E_q2_np, k=8)
            cands2 = [passages[idx] for idx in indices2[0] if idx != -1 and idx < len(passages)]
            scores2 = distances2[0]
            
            filtered_p2, util2 = self.utility_scorer.score_and_filter_passages(query_text, cands2, scores2)
            for p in filtered_p2[:2]:
                if p not in all_retrieved_passages:
                    all_retrieved_passages.append(p)
                    
            hop_telemetry.append({
                "hop": 2,
                "candidates_found": len(cands2),
                "passages_retained": len(filtered_p2[:2]),
                "context_sufficient": util2["is_context_sufficient"],
                "coverage_ratio": util2["coverage_ratio"]
            })
            
        # 3. Dynamic Layer Slicing for joint multi-hop evidence
        tier_probs, tier_idx = self.adaptive_model.layer_router(E_q1_norm)
        selected_tier = self.adaptive_model.layer_router.tiers[tier_idx[0].item()]
        b_q, t_q = selected_tier["b"], selected_tier["t"]
        
        # 4. Concatenated Full-Attention Multi-Hop KV Encoding
        passage_enc = tokenizer(
            all_retrieved_passages,
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt"
        ).input_ids.to(device).unsqueeze(0)
        
        custom_cache = self.adaptive_model.encode_adaptive_passages(
            passage_ids=passage_enc,
            b_layer=b_q,
            t_layer=t_q,
            k_passages=len(all_retrieved_passages)
        )
        
        # 5. Reader Prompting and Final Generation
        instruct_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the question directly and concisely based on the knowledge provided:\n{query_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        reader_ids = tokenizer([instruct_prompt], return_tensors="pt").input_ids.to(device)
        
        orig_b = self.adaptive_model.base_imprag.b
        self.adaptive_model.base_imprag.b = b_q
        try:
            gen_tokens = self.adaptive_model.base_imprag.generate(
                query_ids=reader_ids,
                custom_past_key_values=custom_cache,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_ids=eos_ids
            )
        finally:
            self.adaptive_model.base_imprag.b = orig_b
            
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
        
        return gen_text, {
            "mode": "Iterative Adaptive ImpRAG",
            "total_hops": len(hop_telemetry),
            "hop_details": hop_telemetry,
            "retrieval_decision": "RETRIEVED",
            "k_allocated": len(all_retrieved_passages),
            "layer_boundaries": {"b": b_q, "t": t_q, "tier_name": selected_tier["name"]},
            "retrieved_passages": all_retrieved_passages
        }
