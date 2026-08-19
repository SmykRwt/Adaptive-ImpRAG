import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from imprag.model import ImpRAGModel
from imprag.retriever import ImpRAGFAISSIndex
from imprag.adaptive import AdaptiveImpRAGModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="ImpRAG & Adaptive ImpRAG Quality Evaluator")
    parser.add_argument("--max_samples", type=int, default=200, help="Number of queries to evaluate")
    parser.add_argument("--mode", type=str, default="both", choices=["baseline", "adaptive", "both"], help="Evaluation mode")
    args = parser.parse_args()

    checkpoint_dir = "imp_rag_checkpoint"
    index_path = "imp_rag_wiki.index"
    corpus_path = "wiki_passages.json"
    dataset_path = "train_dataset_with_pseudo_labels.json"
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(index_path) or not os.path.exists(corpus_path) or not os.path.exists(dataset_path):
        print("Error: Missing required files in workspace directory.")
        return
        
    print("=" * 80)
    print("Adaptive ImpRAG vs Baseline ImpRAG - Retrieval & Quality Evaluator")
    print("=" * 80)
    
    print("Loading tokenizer, model, and index...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
        low_cpu_mem_usage=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    num_layers = base_model.config.num_hidden_layers
    b, t = (7, 23) if num_layers == 32 else (7, 19)
    
    baseline_model = ImpRAGModel(base_model, b=b, t=t, pooling_type="last_token").to(device)
    baseline_model.eval()
    
    adaptive_model = AdaptiveImpRAGModel(base_model, default_b=b, default_t=t, pooling_type="last_token").to(device)
    adaptive_model.eval()
    
    faiss_index = ImpRAGFAISSIndex.load(index_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        passages = json.load(f)
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print("Building fast passage index map...")
    passage_to_idx = {p.strip().lower(): idx for idx, p in enumerate(passages)}
    
    eval_split_size = int(len(dataset) * 0.2)
    eval_data = dataset[-eval_split_size:]
    if args.max_samples and args.max_samples < len(eval_data):
        eval_data = eval_data[:args.max_samples]
    print(f"Evaluating on {len(eval_data)} queries...")
    
    # 1. Baseline Evaluation
    if args.mode in ["baseline", "both"]:
        print("\n" + "-" * 40)
        print("Running Baseline ImpRAG Evaluation (Fixed b=7, t=23, Fixed k=5)...")
        recalls = {1: 0, 5: 0, 10: 0}
        mrr_sum = 0.0
        total_valid = 0
        
        with torch.no_grad():
            for item in tqdm(eval_data, desc="Baseline Eval"):
                query_text = item["query"]
                pos_texts = item.get("positive_passages", [])
                pos_indices = [passage_to_idx[p.strip().lower()] for p in pos_texts if p.strip().lower() in passage_to_idx]
                if not pos_indices:
                    continue
                    
                total_valid += 1
                q_enc = tokenizer([query_text], return_tensors="pt")
                E_q = baseline_model.get_retriever_embeddings(q_enc.input_ids.to(device), q_enc.attention_mask.to(device), is_query=True)
                E_q_norm = F.normalize(E_q, p=2, dim=-1)[0].float().cpu().numpy()
                
                distances, indices = faiss_index.search(np.expand_dims(E_q_norm, axis=0), k=10)
                retrieved_ids = indices[0].tolist()
                
                found_rank = None
                for rank, idx in enumerate(retrieved_ids, 1):
                    if idx in pos_indices:
                        found_rank = rank
                        break
                if found_rank is not None:
                    mrr_sum += 1.0 / found_rank
                    if found_rank <= 1: recalls[1] += 1
                    if found_rank <= 5: recalls[5] += 1
                    if found_rank <= 10: recalls[10] += 1
                    
        print("\n--- Baseline ImpRAG Results ---")
        if total_valid > 0:
            print(f"Valid Queries: {total_valid}")
            print(f"Recall@1:  {recalls[1] / total_valid * 100:.2f}%")
            print(f"Recall@5:  {recalls[5] / total_valid * 100:.2f}%")
            print(f"Recall@10: {recalls[10] / total_valid * 100:.2f}%")
            print(f"MRR:       {mrr_sum / total_valid:.4f}")
            print(f"Average k: 5.0 passages (Static)")
            
    # 2. Adaptive Evaluation
    if args.mode in ["adaptive", "both"]:
        print("\n" + "-" * 40)
        print("Running Adaptive ImpRAG Evaluation (4D Dynamic Adaptivity)...")
        recalls_ad = {1: 0, 5: 0, 10: 0}
        mrr_sum_ad = 0.0
        total_valid_ad = 0
        total_k_allocated = 0
        total_bypassed = 0
        tier_counts = {"Shallow": 0, "Standard": 0, "Deep": 0}
        
        with torch.no_grad():
            for item in tqdm(eval_data, desc="Adaptive Eval"):
                query_text = item["query"]
                pos_texts = item.get("positive_passages", [])
                pos_indices = [passage_to_idx[p.strip().lower()] for p in pos_texts if p.strip().lower() in passage_to_idx]
                if not pos_indices:
                    continue
                    
                total_valid_ad += 1
                q_enc = tokenizer([query_text], return_tensors="pt")
                q_ids = q_enc.input_ids.to(device)
                
                # Adaptive query embedding with GQA head attention
                E_q, head_weights = adaptive_model.get_adaptive_query_embedding(q_ids)
                E_q_norm = F.normalize(E_q, p=2, dim=-1)
                E_q_np = E_q_norm.float().cpu().numpy()
                
                # Search FAISS
                distances, indices = faiss_index.search(E_q_np, k=10)
                raw_scores = distances[0]
                retrieved_ids = indices[0].tolist()
                
                # Dynamic k allocation
                k_opt, k_meta = adaptive_model.k_allocator.compute_k(raw_scores)
                total_k_allocated += k_opt
                
                # Layer boundary tier
                tier_probs, tier_idx = adaptive_model.layer_router(E_q_norm)
                tier_name = adaptive_model.layer_router.tiers[tier_idx[0].item()]["name"].split()[0]
                tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
                
                found_rank = None
                for rank, idx in enumerate(retrieved_ids[:k_opt], 1):
                    if idx in pos_indices:
                        found_rank = rank
                        break
                if found_rank is not None:
                    mrr_sum_ad += 1.0 / found_rank
                    if found_rank <= 1: recalls_ad[1] += 1
                    if found_rank <= 5: recalls_ad[5] += 1
                    if found_rank <= 10: recalls_ad[10] += 1
                    
        print("\n--- Adaptive ImpRAG Results ---")
        if total_valid_ad > 0:
            avg_k = total_k_allocated / total_valid_ad
            compute_reduction = max(0.0, (5.0 - avg_k) / 5.0 * 100)
            print(f"Valid Queries: {total_valid_ad}")
            print(f"Recall@1:  {recalls_ad[1] / total_valid_ad * 100:.2f}%")
            print(f"Recall@5:  {recalls_ad[5] / total_valid_ad * 100:.2f}%")
            print(f"Recall@10: {recalls_ad[10] / total_valid_ad * 100:.2f}%")
            print(f"MRR:       {mrr_sum_ad / total_valid_ad:.4f}")
            print(f"Average k Allocated: {avg_k:.2f} passages (Dynamic)")
            print(f"Average KV Cache Compute Savings: {compute_reduction:.1f}%")
            print(f"Layer Boundary Tier Distribution: {tier_counts}")
    print("=" * 80)

if __name__ == "__main__":
    main()
