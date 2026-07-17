import json
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import faiss

def mean_pooling(token_embeddings, attention_mask):
    # Mean pooling for Contriever embeddings
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
    return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

def embed_texts(texts, tokenizer, model, batch_size=32, device="cpu"):
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs.last_hidden_state, inputs.attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0)

def main():
    print("=" * 60)
    print("Pseudo-Label Generation using Contriever MSMARCO")
    print("=" * 60)
    
    # 1. Load Wikipedia passages
    print("Loading Wikipedia passages...")
    with open("wiki_passages.json", "r", encoding="utf-8") as f:
        passages = json.load(f)
    print(f"Loaded {len(passages)} passages.")
    
    # 2. Load Contriever model
    print("Loading Contriever MSMARCO model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    model = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
    
    # 3. Embed Wikipedia corpus
    print("Embedding Wikipedia corpus (this will take 1-2 minutes on CPU)...")
    # Take a subset of 10,000 passages to fit in memory and run fast on CPU
    max_passages = min(10000, len(passages))
    passages = passages[:max_passages]
    
    corpus_embeddings = embed_texts(passages, tokenizer, model, batch_size=32, device=device)
    
    # 4. Build FAISS index for Contriever
    print("Building FAISS index for corpus...")
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings.astype(np.float32))
    
    # 5. Load QA datasets
    print("Loading Natural Questions (NQ-Open) and HotpotQA...")
    # Load subsets of train splits (1000 each) via streaming to avoid large downloads
    try:
        nq_stream = load_dataset("nq_open", split="train", streaming=True)
        nq_dataset = list(nq_stream.take(1000))
    except Exception as e:
        print("Fallback: loading nq_open failed:", str(e))
        nq_dataset = [{"question": "what is the capital of france", "answer": ["Paris"]}]
        
    try:
        hotpot_stream = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
        hotpot_dataset = list(hotpot_stream.take(1000))
    except Exception as e:
        print("Fallback: loading hotpotqa/hotpot_qa failed:", str(e))
        hotpot_dataset = [{"question": "Who wrote the play Hamlet?", "answer": "William Shakespeare"}]
        
    # 6. Retrieve pseudo-labels for training queries
    print("Retrieving top pseudo-positives and hard negatives for queries...")
    dataset_with_labels = []
    
    # Process NQ
    for item in tqdm(nq_dataset, desc="Processing NQ"):
        question = item["question"]
        # Format query prompt
        query = f"Q: {question} A: "
        # Extract answers
        ans_list = item.get("answer", ["No answer"])
        answer = ans_list[0] if isinstance(ans_list, list) else ans_list
        
        # Encode question
        q_emb = embed_texts([question], tokenizer, model, device=device)
        
        # Search top 50 in index
        distances, indices = index.search(q_emb.astype(np.float32), 50)
        indices = indices[0].tolist()
        
        # Positives are top 5
        positives = [passages[idx] for idx in indices[:5] if idx != -1]
        
        # Negatives are sampled from rank 10-50
        neg_candidates = [passages[idx] for idx in indices[10:50] if idx != -1]
        if len(neg_candidates) < 5:
            neg_candidates = passages  # Fallback to random passages
        negatives = random.sample(neg_candidates, min(5, len(neg_candidates)))
        
        dataset_with_labels.append({
            "query": query,
            "answer": answer,
            "positive_passages": positives,
            "negative_passages": negatives
        })
        
    # Process HotpotQA
    for item in tqdm(hotpot_dataset, desc="Processing HotpotQA"):
        question = item["question"]
        query = f"Q: {question} A: "
        answer = item["answer"]
        
        q_emb = embed_texts([question], tokenizer, model, device=device)
        distances, indices = index.search(q_emb.astype(np.float32), 50)
        indices = indices[0].tolist()
        
        positives = [passages[idx] for idx in indices[:5] if idx != -1]
        neg_candidates = [passages[idx] for idx in indices[10:50] if idx != -1]
        if len(neg_candidates) < 5:
            neg_candidates = passages
        negatives = random.sample(neg_candidates, min(5, len(neg_candidates)))
        
        dataset_with_labels.append({
            "query": query,
            "answer": answer,
            "positive_passages": positives,
            "negative_passages": negatives
        })
        
    # Save the labeled dataset
    output_path = "train_dataset_with_pseudo_labels.json"
    print(f"Saving labeled dataset to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_with_labels, f, ensure_ascii=False, indent=2)
        
    print("Pseudo-label generation complete!")

if __name__ == "__main__":
    main()
