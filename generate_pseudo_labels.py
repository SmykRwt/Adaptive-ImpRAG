import json
import random
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import faiss

def mean_pooling(token_embeddings, attention_mask):
    # Mean pooling for Contriever embeddings
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.)
    return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

def embed_texts(texts, tokenizer, model, batch_size=512, device="cpu"):
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs.last_hidden_state, inputs.attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Pseudo-Label Generation using Contriever MSMARCO")
    parser.add_argument("--max_queries", type=int, default=25000, help="Max queries to take from NQ and HotpotQA each (total = 2 * max_queries)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for embedding queries")
    parser.add_argument("--corpus_batch_size", type=int, default=512, help="Batch size for embedding corpus passages")
    args = parser.parse_args()

    print("=" * 60)
    print("Pseudo-Label Generation using Contriever MSMARCO (GPU Optimized)")
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
    print(f"Embedding all {len(passages)} Wikipedia corpus passages...")
    corpus_embeddings = embed_texts(passages, tokenizer, model, batch_size=args.corpus_batch_size, device=device)
    
    # 4. Build FAISS index for Contriever
    print("Building FAISS index for corpus...")
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings.astype(np.float32))
    
    if device == "cuda":
        print("Transferring FAISS index to GPU...")
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("Successfully moved FAISS index to GPU.")
        except Exception as e:
            print("Warning: Could not move index to GPU, using CPU index instead:", str(e))
    
    # 5. Load QA datasets
    print(f"Loading up to {args.max_queries} samples from NQ-Open and HotpotQA...")
    try:
        nq_stream = load_dataset("nq_open", split="train", streaming=True)
        nq_dataset = list(nq_stream.take(args.max_queries))
        print(f"Loaded {len(nq_dataset)} NQ queries.")
    except Exception as e:
        print("Fallback: loading nq_open failed:", str(e))
        nq_dataset = [{"question": "what is the capital of france", "answer": ["Paris"]}]
        
    try:
        hotpot_stream = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
        hotpot_dataset = list(hotpot_stream.take(args.max_queries))
        print(f"Loaded {len(hotpot_dataset)} HotpotQA queries.")
    except Exception as e:
        print("Fallback: loading hotpotqa/hotpot_qa failed:", str(e))
        hotpot_dataset = [{"question": "Who wrote the play Hamlet?", "answer": "William Shakespeare"}]
        
    # 6. Batched query retrieval
    print("\nBatch embedding all queries...")
    all_queries = []
    
    # Collect NQ
    for item in nq_dataset:
        question = item["question"]
        ans_list = item.get("answer", ["No answer"])
        answer = ans_list[0] if isinstance(ans_list, list) else ans_list
        all_queries.append({
            "question": question,
            "query": f"Q: {question} A: ",
            "answer": answer,
            "source": "NQ"
        })
        
    # Collect HotpotQA
    for item in hotpot_dataset:
        question = item["question"]
        answer = item["answer"]
        all_queries.append({
            "question": question,
            "query": f"Q: {question} A: ",
            "answer": answer,
            "source": "HotpotQA"
        })
        
    questions = [q["question"] for q in all_queries]
    
    print(f"Embedding {len(questions)} queries...")
    query_embeddings = embed_texts(questions, tokenizer, model, batch_size=args.batch_size, device=device)
    
    print("Performing batched FAISS search for pseudo-labels...")
    distances, indices = index.search(query_embeddings.astype(np.float32), 50)
    
    # Process search results
    print("Assembling positive and negative passages...")
    dataset_with_labels = []
    
    for i in tqdm(range(len(all_queries)), desc="Processing search results"):
        query_info = all_queries[i]
        q_indices = indices[i].tolist()
        
        # Positives are top 5
        positives = [passages[idx] for idx in q_indices[:5] if idx != -1]
        
        # Negatives are sampled from rank 10-50
        neg_candidates = [passages[idx] for idx in q_indices[10:50] if idx != -1]
        if len(neg_candidates) < 5:
            neg_candidates = passages
            
        negatives = random.sample(neg_candidates, min(5, len(neg_candidates)))
        
        dataset_with_labels.append({
            "query": query_info["query"],
            "answer": query_info["answer"],
            "positive_passages": positives,
            "negative_passages": negatives
        })
        
    # Save the labeled dataset
    output_path = "train_dataset_with_pseudo_labels.json"
    print(f"Saving {len(dataset_with_labels)} labeled queries to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_with_labels, f, ensure_ascii=False, indent=2)
        
    print("Pseudo-label generation complete!")

if __name__ == "__main__":
    main()
