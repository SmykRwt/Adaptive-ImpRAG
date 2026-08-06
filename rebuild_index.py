import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from imprag.model import ImpRAGModel
from imprag.retriever import ImpRAGFAISSIndex
import numpy as np
from tqdm import tqdm

def main():
    print("=" * 60)
    print("Rebuilding FAISS Retrieval Index Locally")
    print("=" * 60)
    
    checkpoint_dir = "imp_rag_checkpoint"
    corpus_path = "wiki_passages.json"
    output_index_path = "imp_rag_wiki.index"
    
    # Verification
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint folder '{checkpoint_dir}' not found.")
        print("Please unzip your 'imp_rag_checkpoint' zip file directly into this folder first.")
        return
        
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus '{corpus_path}' not found. Run prepare_wiki_corpus.py first.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Wikipedia passages
    print("Loading Wikipedia passages...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        passages = json.load(f)
    print(f"Loaded {len(passages)} passages.")
    
    # 2. Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if local weights exist, otherwise download from Hugging Face Hub
    weights_exist = False
    for fname in ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"]:
        if os.path.exists(os.path.join(checkpoint_dir, fname)):
            weights_exist = True
            break
            
    model_source = checkpoint_dir if weights_exist else "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Loading base model from: {model_source}...")
    
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    
    # Wrap model (b=7 for Qwen 28-layer config)
    model = ImpRAGModel(base_model, b=7, t=19, k_passages=2, max_passage_len=64)
    model.to(device)
    model.eval()
    
    # 3. Index corpus
    print("Embedding Wikipedia corpus (using bottom 8 layers only)...")
    all_embeddings = []
    dimension = None
    batch_size = 128 if device == "cuda" else 16
    
    # Temporary patch to only run layers 0..b to save CPU time
    original_layers = model.base_model.model.layers
    model.base_model.model.layers = original_layers[:model.b + 1]
    
    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(passages), batch_size), desc="FAISS Indexing"):
                batch_texts = passages[i : i + batch_size]
                encoding = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                passage_ids = encoding.input_ids.to(device)
                passage_mask = encoding.attention_mask.to(device)
                
                # Extract passage embeddings using the hidden states
                emb = model.get_retriever_embeddings(passage_ids, attention_mask=passage_mask, is_query=False)
                
                if dimension is None:
                    dimension = emb.shape[-1]
                    
                all_embeddings.append(emb.float().cpu().numpy())
    finally:
        model.base_model.model.layers = original_layers
        
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # 4. Create and Save FAISS Index
    print(f"Building FAISS Index with {len(all_embeddings)} vectors (dimension={dimension})...")
    faiss_index = ImpRAGFAISSIndex(dimension)
    faiss_index.add_embeddings(all_embeddings)
    
    faiss_index.save(output_index_path)
    print("Success! FAISS index and mean vector rebuilt.")
    
    # 3. Compute and save query mean vector for dual-centering
    print("Computing query mean vector for dual-centering...")
    if os.path.exists(corpus_path):
        # We can extract queries from the training split
        train_path = "train_dataset_with_pseudo_labels.json"
        if os.path.exists(train_path):
            with open(train_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            
            query_vectors = []
            with torch.no_grad():
                for item in train_data[:100]:
                    q_enc = tokenizer([item["query"]], return_tensors="pt")
                    q_ids = q_enc.input_ids.to(device)
                    q_mask = q_enc.attention_mask.to(device)
                    
                    E_q = model.get_retriever_embeddings(q_ids, attention_mask=q_mask, is_query=True)
                    E_q_norm = F.normalize(E_q, p=2, dim=-1)[0].float().cpu().numpy()
                    query_vectors.append(E_q_norm)
                    
            query_vectors = np.array(query_vectors, dtype=np.float32)
            mean_q = query_vectors.mean(axis=0, keepdims=True)
            
            # Save using the retriever's expected filename
            np.save(output_index_path + ".query_mean.npy", mean_q)
            print("Success! Query mean vector generated and saved.")
        else:
            print("Warning: train_dataset_with_pseudo_labels.json not found, skipping query mean.")
    else:
        print("Warning: Wikipedia corpus not found, skipping query mean.")
    print(f"\nSuccess! FAISS index rebuilt and saved to: {output_index_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
