import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from imprag.model import ImpRAGModel
from imprag.dataset import ImpRAGDataset, collate_fn
from imprag.trainer import ImpRAGTrainer
from imprag.retriever import ImpRAGFAISSIndex
import numpy as np
from tqdm import tqdm
import os

def index_corpus(model, tokenizer, passages, batch_size=16, device="cpu"):
    """
    Embeds the entire corpus using the model's bottom group L_B
    and indexes them in a FAISS index.
    """
    model.eval()
    all_embeddings = []
    dimension = None
    
    with torch.no_grad():
        for i in tqdm(range(0, len(passages), batch_size), desc="FAISS Corpus Indexing"):
            batch_texts = passages[i : i + batch_size]
            encoding = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            passage_ids = encoding.input_ids.to(device)
            passage_mask = encoding.attention_mask.to(device)
            
            # Extract passage embeddings using the key projections at layer b
            emb = model.get_retriever_embeddings(passage_ids, attention_mask=passage_mask, is_query=False)
            
            # Normalize embeddings to unit vectors for cosine similarity
            emb_norm = F.normalize(emb, p=2, dim=-1)
            
            if dimension is None:
                dimension = emb_norm.shape[-1]
                
            all_embeddings.append(emb_norm.float().cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Initialize and populate the FAISS index
    faiss_index = ImpRAGFAISSIndex(dimension)
    faiss_index.add_embeddings(all_embeddings)
    return faiss_index, all_embeddings

def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print("=" * 60)
    print(f"ImpRAG Baseline Pipeline - {model_name.split('/')[-1]}")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load passages and train dataset
    if not os.path.exists("wiki_passages.json"):
        raise FileNotFoundError("wiki_passages.json not found. Run prepare_wiki_corpus.py first.")
        
    if not os.path.exists("train_dataset_with_pseudo_labels.json"):
        raise FileNotFoundError("train_dataset_with_pseudo_labels.json not found. Run generate_pseudo_labels.py first.")
        
    with open("wiki_passages.json", "r", encoding="utf-8") as f:
        passages = json.load(f)
        
    with open("train_dataset_with_pseudo_labels.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    print(f"Loaded {len(passages)} corpus passages.")
    print(f"Loaded {len(train_data)} training queries with pseudo-labels.")
    
    # Split training data into train (80%) and eval (20%)
    split_idx = int(len(train_data) * 0.8)
    train_split = train_data[:split_idx]
    eval_split = train_data[split_idx:]
    
    print(f"Train split size: {len(train_split)}, Eval split size: {len(eval_split)}")
    
    # 2. Load model and tokenizer
    print("Loading pre-trained Qwen2.5-1.5B-Instruct model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Keep right padding for labels mapping simplicity and use our mask-based pooling
    tokenizer.padding_side = "right"
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    # 3. Slice and wrap the model
    # Qwen-2.5-1.5B has 28 layers. We set boundaries matching the paper's 28-layer setup:
    # b = 7 (bottom layers 0..7)
    # t = 19 (middle layers 7..19)
    # top group LT layers 20..27
    model = ImpRAGModel(base_model, b=7, t=19, k_passages=2, max_passage_len=64)
    model.to(device)
    
    # 4. Create Datasets and DataLoaders
    train_dataset = ImpRAGDataset(train_split)
    eval_dataset = ImpRAGDataset(eval_split)
    
    # Smaller batch size to prevent memory warnings/out-of-memory on CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_query_len=64, max_passage_len=64, num_candidates=5)
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_query_len=64, max_passage_len=64, num_candidates=5)
    )
    
    # 5. Initialize Optimizer and Trainer
    from transformers.optimization import Adafactor
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=3e-5)
    
    # 6 epochs total: 2 Warmup (NCE), 4 Distillation (KL)
    trainer = ImpRAGTrainer(
        model=model,
        optimizer=optimizer,
        warmup_epochs=2,
        total_epochs=6,
        lambda_ret=1.0,
        device=device
    )
    
    # 6. Evaluation BEFORE training
    print("\nEvaluating Model BEFORE Training...")
    # Evaluate with standard loader (compares to query candidates)
    pre_em, pre_recall = trainer.evaluate(eval_loader, tokenizer)
    print(f"Pre-training Candidate EM: {pre_em*100:.2f}%")
    print(f"Pre-training Candidate Retrieval Recall: {pre_recall*100:.2f}%")
    
    # 7. Run Two-Stage Training
    print("\nStarting Two-Stage Training Loop...")
    for epoch in range(trainer.total_epochs):
        loss, gen_loss, ret_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch+1}/{trainer.total_epochs} - Completed. Avg Loss: {loss:.4f} (Gen: {gen_loss:.4f}, Ret: {ret_loss:.4f})\n")
        
    # 8. Index Corpus post-training using trained bottom layers (L_B)
    print("\nBuilding post-training FAISS Retrieval Index...")
    faiss_index, corpus_embs = index_corpus(model, tokenizer, passages, batch_size=16, device=device)
    print(f"Successfully built FAISS index with {faiss_index.get_num_vectors()} vectors.")
    
    # Save the FAISS index
    faiss_index.save("imp_rag_wiki.index")
    print("FAISS index saved to imp_rag_wiki.index.")
    
    # 9. Evaluate AFTER training using the FAISS Index
    print("\nEvaluating Model AFTER Training (Full Corpus Retrieval via FAISS)...")
    exact_matches = 0
    retrieval_recalls = 0
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for item in tqdm(eval_split, desc="Evaluating on Eval Split"):
            query_text = item["query"]
            answer_text = item["answer"].strip().lower()
            
            # Embed query prompt
            q_enc = tokenizer([query_text], return_tensors="pt")
            q_ids = q_enc.input_ids.to(device)
            q_mask = q_enc.attention_mask.to(device)
            
            E_q = model.get_retriever_embeddings(q_ids, attention_mask=q_mask, is_query=True)
            # L2-normalize query embedding
            E_q_norm = F.normalize(E_q, p=2, dim=-1)
            E_q_np = E_q_norm.float().cpu().numpy()
            
            # Query FAISS index for top-5 documents
            distances, indices = faiss_index.search(E_q_np, k=5)
            ret_passages = [passages[idx] for idx in indices[0] if idx != -1]
            
            # 1. Retrieval Recall: check if answer is a substring in retrieved passages
            has_answer = False
            for p in ret_passages:
                if answer_text in p.lower():
                    has_answer = True
                    break
            if has_answer:
                retrieval_recalls += 1
                
            # 2. Generation EM
            # Encode retrieved top-2 passages
            retrieved_passage_ids = tokenizer(ret_passages[:2], padding=True, truncation=True, max_length=64, return_tensors="pt").input_ids.to(device)
            custom_cache = model.encode_passages(retrieved_passage_ids)
            
            # Generate answer
            gen_tokens = model.generate(q_ids, custom_cache, max_new_tokens=15)
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip().lower()
            
            # Check Exact Match
            if answer_text in gen_text or gen_text == answer_text:
                exact_matches += 1
                
            total_samples += 1
            
    final_em = exact_matches / total_samples
    final_recall = retrieval_recalls / total_samples
    
    print("\n" + "=" * 40)
    print("Final Evaluation Results (Full FAISS Corpus Retrieval)")
    print(f"Exact Match (EM): {final_em*100:.2f}%")
    print(f"Retrieval Recall: {final_recall*100:.2f}%")
    print("=" * 40 + "\n")
    
    # 10. Save Model Checkpoint
    checkpoint_dir = "imp_rag_checkpoint"
    print(f"Saving final trained checkpoint to {checkpoint_dir}...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print("Checkpoint saved successfully!")
    
    # Auto-copy to Google Drive if running in Colab
    if os.path.exists("/content/drive/MyDrive"):
        import shutil
        print("Google Drive connection detected! Backing up files to Google Drive root automatically...")
        try:
            # Copy checkpoint folder
            drive_checkpoint_dir = "/content/drive/MyDrive/imp_rag_checkpoint"
            if os.path.exists(drive_checkpoint_dir):
                shutil.rmtree(drive_checkpoint_dir)
            shutil.copytree(checkpoint_dir, drive_checkpoint_dir)
            print("Checkpoint copied to Google Drive successfully!")
            
            # Copy FAISS index
            shutil.copy("imp_rag_wiki.index", "/content/drive/MyDrive/imp_rag_wiki.index")
            print("FAISS Index copied to Google Drive successfully!")
            print("All assets backed up to your Google Drive root folder!")
        except Exception as e:
            print(f"Warning: Failed to copy to Google Drive automatically: {str(e)}")
            
    print("\nImpRAG pipeline completed successfully!")

if __name__ == "__main__":
    main()
