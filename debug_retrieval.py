import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from imprag.model import ImpRAGModel
from imprag.retriever import ImpRAGFAISSIndex

def check_raw_corpus_coverage(corpus, keywords):
    """
    Scans the corpus for any occurrences of keywords.
    """
    matches = []
    for idx, text in enumerate(corpus):
        if any(kw.lower() in text.lower() for kw in keywords):
            matches.append((idx, text))
    return matches

def main():
    checkpoint_dir = "imp_rag_checkpoint"
    index_path = "imp_rag_wiki.index"
    corpus_path = "wiki_passages.json"
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(index_path) or not os.path.exists(corpus_path):
        print("Error: Missing required files in the workspace directory.")
        return
        
    print("=" * 80)
    print("ImpRAG Retrieval Diagnostics & Debugger")
    print("=" * 80)
    
    # 1. Load tokenizer, model, and index
    print("Loading models and index...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    # Slice and wrap the model dynamically based on number of layers (Section 4.1 in paper)
    num_layers = base_model.config.num_hidden_layers
    if num_layers == 32:
        b = 7
        t = 23
    elif num_layers == 28:
        b = 7
        t = 19
    else:
        b = 7
        t = int(num_layers * 0.75)
        print(f"Warning: Unexpected layer count {num_layers}. Slicing: b={b}, t={t}")
        
    model = ImpRAGModel(base_model, b=b, t=t, pooling_type="last_token")
    model.eval()
    
    faiss_index = ImpRAGFAISSIndex.load(index_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        passages = json.load(f)
        
    # Get query from command line arguments or use default
    query = "where did they film hot tub time machine"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
    print(f"\nTarget Query: '{query}'")
    
    # 2. Check raw corpus coverage
    # Extract keywords (ignoring short stopwords)
    stopwords = {"where", "did", "they", "who", "what", "is", "the", "in", "of", "a", "for", "on", "was", "how", "to", "first"}
    keywords = [word.strip('?,.!"\'') for word in query.split() if word.lower() not in stopwords and len(word) > 2]
    
    print(f"Keywords for raw scan: {keywords}")
    matches = check_raw_corpus_coverage(passages, keywords)
    
    print("\n--- 1. Corpus Coverage Check ---")
    if matches:
        print(f"Found {len(matches)} matching passages in raw simple Wikipedia corpus:")
        for idx, text in matches[:3]:
            # Print safe ASCII to prevent encoding errors
            safe_text = text[:150].encode('ascii', 'ignore').decode('ascii')
            print(f"  [Index {idx}]: {safe_text}...")
        if len(matches) > 3:
            print(f"  ... and {len(matches) - 3} more.")
    else:
        print("Keyword NOT found in raw wikitext-2 corpus passages!")
        print("  Meaning: The factual answer does NOT exist in this dataset. (Corpus Scope Deficit)")
        
    # 3. Compute query embedding
    print("\n--- 2. Query Embedding Diagnostics ---")
    formatted_query = f"Q: {query} A: "
    q_enc = tokenizer([formatted_query], return_tensors="pt")
    
    with torch.no_grad():
        E_q = model.get_retriever_embeddings(q_enc.input_ids, q_enc.attention_mask, is_query=True)
        E_q_norm = F.normalize(E_q, p=2, dim=-1)[0].float().cpu().numpy()
        
    print(f"Query representation dimension: {len(E_q_norm)}")
    print(f"Query embedding L2 norm: {np.linalg.norm(E_q_norm):.6f}")
    
    # 4. Search FAISS Index
    print("\n--- 3. Top-10 Retrieval Rankings ---")
    # Wrap in expand_dims to search
    distances, indices = faiss_index.search(np.expand_dims(E_q_norm, axis=0), k=10)
    
    # Print table-like listing
    print(f"{'Rank':<5} | {'Doc ID':<8} | {'Similarity Score':<18} | {'Passage Preview (First 100 chars)'}")
    print("-" * 100)
    for rank, (idx, dist) in enumerate(zip(indices[0].tolist(), distances[0].tolist()), 1):
        if idx != -1:
            safe_text = passages[idx][:100].encode('ascii', 'ignore').decode('ascii').replace('\n', ' ')
            print(f"{rank:<5} | {idx:<8} | {dist:<18.6f} | {safe_text}...")
        else:
            print(f"{rank:<5} | {'None':<8} | {0.0:<18.6f} | -")

if __name__ == "__main__":
    main()
