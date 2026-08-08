import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from imprag.model import ImpRAGModel
from imprag.retriever import ImpRAGFAISSIndex
from tqdm import tqdm

def main():
    checkpoint_dir = "imp_rag_checkpoint"
    index_path = "imp_rag_wiki.index"
    corpus_path = "wiki_passages.json"
    dataset_path = "train_dataset_with_pseudo_labels.json"
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(index_path) or not os.path.exists(corpus_path) or not os.path.exists(dataset_path):
        print("Error: Missing required files.")
        return
        
    print("=" * 80)
    print("ImpRAG Retrieval Quality Evaluator")
    print("=" * 80)
    
    # 1. Load resources
    print("Loading models and index...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, 
        torch_dtype=model_dtype, 
        low_cpu_mem_usage=True
    )
    # Slice and wrap the model dynamically based on number of layers
    num_layers = base_model.config.num_hidden_layers
    if num_layers == 32:
        b = 15
        t = 23
    elif num_layers == 28:
        b = 14
        t = 19
    else:
        b = int(num_layers * 0.5)
        t = int(num_layers * 0.7)
        print(f"Warning: Unexpected layer count {num_layers}. Slicing: b={b}, t={t}")
        
    model = ImpRAGModel(base_model, b=b, t=t, pooling_type="mean")
    model.eval()
    
    faiss_index = ImpRAGFAISSIndex.load(index_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        passages = json.load(f)
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # Map raw passage texts to their indices in simple Wikipedia corpus
    print("Building passage index map...")
    passage_to_idx = {p.strip().lower(): idx for idx, p in enumerate(passages)}
    
    # Split: Take the last 20% of the dataset as the evaluation split (e.g. 200 queries)
    eval_split_size = int(len(dataset) * 0.2)
    eval_data = dataset[-eval_split_size:]
    print(f"Loaded {len(eval_data)} queries for evaluation.")
    
    # 2. Evaluation Loop
    recalls = {1: 0, 5: 0, 10: 0}
    mrr_sum = 0.0
    total_valid_queries = 0
    
    with torch.no_grad():
        for item in tqdm(eval_data, desc="Evaluating Queries"):
            query_text = item["query"]
            pos_texts = item.get("positive_passages", [])
            
            # Find the indices of the positive passages in our corpus
            pos_indices = []
            for p_txt in pos_texts:
                p_txt_clean = p_txt.strip().lower()
                if p_txt_clean in passage_to_idx:
                    pos_indices.append(passage_to_idx[p_txt_clean])
            
            # If the positive passage is not in the Wikitext-2 corpus,
            # this query has a corpus scope deficit. We skip it from retriever
            # algorithm metrics (since it's impossible for any retriever to fetch it).
            if not pos_indices:
                continue
                
            total_valid_queries += 1
            
            # Embed query
            q_enc = tokenizer([query_text], return_tensors="pt")
            E_q = model.get_retriever_embeddings(q_enc.input_ids.to(base_model.device), q_enc.attention_mask.to(base_model.device), is_query=True)
            E_q_norm = F.normalize(E_q, p=2, dim=-1)[0].float().cpu().numpy()
            
            # Search FAISS Index
            distances, indices = faiss_index.search(np.expand_dims(E_q_norm, axis=0), k=10)
            retrieved_ids = indices[0].tolist()
            
            # Calculate metrics
            found_rank = None
            for rank, idx in enumerate(retrieved_ids, 1):
                if idx in pos_indices:
                    found_rank = rank
                    break
                    
            if found_rank is not None:
                mrr_sum += 1.0 / found_rank
                if found_rank <= 1:
                    recalls[1] += 1
                if found_rank <= 5:
                    recalls[5] += 1
                if found_rank <= 10:
                    recalls[10] += 1
                    
    # 3. Print Results
    print("\n" + "=" * 50)
    print("FINAL RETRIEVAL QUALITY METRICS")
    print("=" * 50)
    if total_valid_queries > 0:
        print(f"Total evaluated queries (with positive passages in corpus): {total_valid_queries}")
        print(f"Recall@1:  {recalls[1] / total_valid_queries * 100:.2f}%")
        print(f"Recall@5:  {recalls[5] / total_valid_queries * 100:.2f}%")
        print(f"Recall@10: {recalls[10] / total_valid_queries * 100:.2f}%")
        print(f"MRR:       {mrr_sum / total_valid_queries:.4f}")
    else:
        print("No queries in the evaluation set had positive passages present in the Wikitext-2 corpus.")
    print("=" * 50)

if __name__ == "__main__":
    main()
