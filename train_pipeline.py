import json
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from imprag.model import ImpRAGModel
from imprag.dataset import ImpRAGDataset, collate_fn
from imprag.trainer import ImpRAGTrainer
from imprag.retriever import ImpRAGFAISSIndex
import numpy as np
from tqdm import tqdm
import os

def index_corpus(model, tokenizer, passages, batch_size=128, device="cpu"):
    """
    Embeds the entire corpus using the model's bottom group L_B
    and indexes them in a FAISS index.
    """
    model.eval()
    all_embeddings = []
    dimension = None
    
    # Handle DDP wrapping
    model_obj = model.module if hasattr(model, "module") else model
    
    original_layers = model_obj.base_model.model.layers
    model_obj.base_model.model.layers = original_layers[:model_obj.b + 1]
    
    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(passages), batch_size), desc="FAISS Corpus Indexing"):
                batch_texts = passages[i : i + batch_size]
                encoding = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                passage_ids = encoding.input_ids.to(device)
                passage_mask = encoding.attention_mask.to(device)
                
                emb = model_obj.get_retriever_embeddings(passage_ids, attention_mask=passage_mask, is_query=False)
                emb_norm = F.normalize(emb, p=2, dim=-1)
                
                if dimension is None:
                    dimension = emb_norm.shape[-1]
                    
                all_embeddings.append(emb_norm.float().cpu().numpy())
    finally:
        model_obj.base_model.model.layers = original_layers
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    faiss_index = ImpRAGFAISSIndex(dimension)
    faiss_index.add_embeddings(all_embeddings)
    return faiss_index, all_embeddings

def main():
    parser = argparse.ArgumentParser(description="ImpRAG Large-Scale GPU Trainer")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model checkpoint to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU forward pass")
    parser.add_argument("--epochs", type=int, default=6, help="Total training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of NCE warmup epochs")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--pooling_type", type=str, default="last_token", choices=["last_token", "mean"], help="Pooling type for retrievals")
    parser.add_argument("--max_train_samples", type=int, default=20000, help="Maximum training queries to use")
    parser.add_argument("--max_eval_samples", type=int, default=2000, help="Maximum evaluation queries to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    args = parser.parse_args()

    # DDP Multi-GPU Initialization
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = args.model
    if local_rank == 0:
        print("=" * 60)
        print(f"ImpRAG GPU Scaled DDP Pipeline - {model_name.split('/')[-1]}")
        print("=" * 60)
        print(f"Using device: {device} (Distributed={is_distributed})")
    
    # 1. Load passages and train dataset
    if not os.path.exists("wiki_passages.json"):
        raise FileNotFoundError("wiki_passages.json not found. Run prepare_wiki_corpus.py first.")
        
    if not os.path.exists("train_dataset_with_pseudo_labels.json"):
        raise FileNotFoundError("train_dataset_with_pseudo_labels.json not found. Run generate_pseudo_labels.py first.")
        
    with open("wiki_passages.json", "r", encoding="utf-8") as f:
        passages = json.load(f)
        
    with open("train_dataset_with_pseudo_labels.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    if local_rank == 0:
        print(f"Loaded {len(passages)} corpus passages.")
        print(f"Loaded {len(train_data)} total training queries with pseudo-labels.")
    
    # Limit samples according to max_train_samples and max_eval_samples
    if args.max_train_samples and args.max_train_samples < len(train_data):
        total_used = args.max_train_samples + (args.max_eval_samples or 1000)
        train_data = train_data[:total_used]
        
    split_idx = int(len(train_data) * 0.8)
    if args.max_train_samples and split_idx > args.max_train_samples:
        split_idx = args.max_train_samples
        
    train_split = train_data[:split_idx]
    eval_split = train_data[split_idx:]
    if args.max_eval_samples and len(eval_split) > args.max_eval_samples:
        eval_split = eval_split[:args.max_eval_samples]
    
    if local_rank == 0:
        print(f"Train split size: {len(train_split)} ({len(train_split) // args.batch_size} batches/epoch), Eval split size: {len(eval_split)}")
    
    # 2. Load model and tokenizer
    if local_rank == 0:
        print(f"Loading pre-trained {model_name} model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # 3. Slice and wrap the model dynamically based on number of layers (Section 4.1 in paper)
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
        if local_rank == 0:
            print(f"Warning: Unexpected layer count {num_layers}. Slicing: b={b}, t={t}")
        
    if local_rank == 0:
        print(f"Model Slicing Boundaries: bottom layer group LB (0..{b}), middle group LM ({b}..{t}), top group LT ({t+1}..{num_layers-1})")
    
    model = ImpRAGModel(base_model, b=b, t=t, k_passages=5, max_passage_len=128, pooling_type=args.pooling_type)
    model.to(device)
    
    # 4. Create Datasets and DDP DataLoaders
    train_dataset = ImpRAGDataset(train_split)
    eval_dataset = ImpRAGDataset(eval_split)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers if device != "cpu" else 0,
        pin_memory=True if device != "cpu" else False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_query_len=64, max_passage_len=128, max_positives_per_query=5, max_negatives_per_query=5)
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=args.num_workers if device != "cpu" else 0,
        pin_memory=True if device != "cpu" else False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_query_len=64, max_passage_len=128, max_positives_per_query=5, max_negatives_per_query=5)
    )
    
    # 5. Initialize Optimizer, Scheduler and Trainer
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only q_proj and k_proj for retriever layers in bottom group LB (0..b)
    unfrozen_count = 0
    for layer_idx in range(b + 1):
        layer = base_model.model.layers[layer_idx]
        for param in layer.self_attn.q_proj.parameters():
            param.requires_grad = True
            unfrozen_count += 1
        for param in layer.self_attn.k_proj.parameters():
            param.requires_grad = True
            unfrozen_count += 1
            
    if local_rank == 0:
        print(f"Paper alignment: Froze all reader/generator weights. Unfroze {unfrozen_count} projection matrices in retriever group LB (Layers 0..{b}).")
    
    # DDP Wrap the model (requires find_unused_parameters because of frozen generator layers)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    from transformers.optimization import Adafactor
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adafactor(trainable_params, scale_parameter=False, relative_step=False, lr=args.lr)
    
    # Setup linear warmup cosine decay scheduler
    total_steps = (len(train_loader) * args.epochs) // args.accumulation_steps
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    if local_rank == 0:
        print(f"Total training steps: {total_steps} (Warmup: {warmup_steps})")
    
    trainer = ImpRAGTrainer(
        model=model,
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        lambda_ret=1.0,
        device=device,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.use_amp,
        scheduler=scheduler
    )
    
    # 6. Evaluation BEFORE training
    if local_rank == 0:
        print("\nEvaluating Model BEFORE Training...")
        pre_em, pre_recall = trainer.evaluate(eval_loader, tokenizer)
        print(f"Pre-training Candidate EM: {pre_em*100:.2f}%")
        print(f"Pre-training Candidate Retrieval Recall: {pre_recall*100:.2f}%")
    
    # 7. Run Two-Stage Training (with automatic checkpoint saving per epoch)
    if local_rank == 0:
        print("\nStarting Two-Stage Training Loop...")
        
    for epoch in range(trainer.total_epochs):
        epoch_ckpt = f"checkpoint_epoch_{epoch + 1}.pt"
        if os.path.exists(epoch_ckpt):
            if local_rank == 0:
                print(f"Loading pre-saved checkpoint for Epoch {epoch + 1} ({epoch_ckpt}). Skipping to next epoch...")
            checkpoint_data = torch.load(epoch_ckpt, map_location=device)
            model_obj = model.module if hasattr(model, "module") else model
            model_obj.load_state_dict(checkpoint_data["model_state"], strict=False)
            continue

        if is_distributed:
            train_sampler.set_epoch(epoch)
        loss, gen_loss, ret_loss = trainer.train_epoch(train_loader, epoch)
        
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{trainer.total_epochs} - Completed. Avg Loss: {loss:.4f} (Gen: {gen_loss:.4f}, Ret: {ret_loss:.4f})\n")
            model_obj = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch + 1,
                "model_state": model_obj.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, epoch_ckpt)
            print(f"Saved Epoch {epoch + 1} checkpoint to {epoch_ckpt}.")
        
    # 8. Index Corpus post-training using trained bottom layers (L_B) on Rank 0
    if local_rank == 0:
        print("\nBuilding post-training FAISS Retrieval Index...")
        faiss_index, corpus_embs = index_corpus(model, tokenizer, passages, batch_size=128, device=device)
        print(f"Successfully built FAISS index with {faiss_index.get_num_vectors()} vectors.")
        
        # Save the FAISS index
        faiss_index.save("imp_rag_wiki.index")
        print("FAISS index saved to imp_rag_wiki.index.")
        
        # Save document mean vector
        p_mean = corpus_embs.mean(axis=0, keepdims=True)
        np.save("imp_rag_wiki.index.mean.npy", p_mean)
        print("Document mean vector saved.")
        
        # Compute and save query mean vector for dual-centering
        print("Computing query mean vector for dual-centering...")
        query_vectors = []
        model.eval()
        model_obj = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            for item in train_data[:100]:
                q_enc = tokenizer([item["query"]], return_tensors="pt")
                E_q = model_obj.get_retriever_embeddings(q_enc.input_ids.to(device), attention_mask=q_enc.attention_mask.to(device), is_query=True)
                E_q_norm = F.normalize(E_q, p=2, dim=-1)[0].float().cpu().numpy()
                query_vectors.append(E_q_norm)
        query_vectors = np.array(query_vectors, dtype=np.float32)
        mean_q = query_vectors.mean(axis=0, keepdims=True)
        np.save("imp_rag_wiki.index.query_mean.npy", mean_q)
        print("Query mean vector saved.")
        
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
                
                q_enc = tokenizer([query_text], return_tensors="pt")
                q_ids = q_enc.input_ids.to(device)
                q_mask = q_enc.attention_mask.to(device)
                
                E_q = model_obj.get_retriever_embeddings(q_ids, attention_mask=q_mask, is_query=True)
                E_q_norm = F.normalize(E_q, p=2, dim=-1)
                E_q_np = E_q_norm.float().cpu().numpy()
                
                # Apply dual-centering
                E_q_centered = E_q_np - p_mean
                E_q_centered /= np.linalg.norm(E_q_centered, axis=-1, keepdims=True)
                
                # Query FAISS index for top-5 documents
                distances, indices = faiss_index.search(E_q_centered, k=5)
                ret_passages = [passages[idx] for idx in indices[0] if idx != -1]
                
                has_answer = False
                for p in ret_passages:
                    if answer_text in p.lower():
                        has_answer = True
                        break
                if has_answer:
                    retrieval_recalls += 1
                    
                retrieved_passage_ids = tokenizer(ret_passages[:5], padding=True, truncation=True, max_length=128, return_tensors="pt").input_ids.to(device).unsqueeze(0)
                custom_cache = model_obj.encode_passages(retrieved_passage_ids, k_passages=len(ret_passages[:5]), encoding_mode="concatenated")
                
                gen_tokens = model_obj.generate(q_ids, custom_cache, max_new_tokens=15)
                gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip().lower()
                
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
        
        if os.path.exists("/content/drive/MyDrive"):
            import shutil
            print("Google Drive connection detected! Backing up files to Google Drive root automatically...")
            try:
                drive_checkpoint_dir = "/content/drive/MyDrive/imp_rag_checkpoint"
                if os.path.exists(drive_checkpoint_dir):
                    shutil.rmtree(drive_checkpoint_dir)
                shutil.copytree(checkpoint_dir, drive_checkpoint_dir)
                shutil.copy("imp_rag_wiki.index", "/content/drive/MyDrive/imp_rag_wiki.index")
                shutil.copy("imp_rag_wiki.index.mean.npy", "/content/drive/MyDrive/imp_rag_wiki.index.mean.npy")
                shutil.copy("imp_rag_wiki.index.query_mean.npy", "/content/drive/MyDrive/imp_rag_wiki.index.query_mean.npy")
                print("All assets backed up to your Google Drive root folder!")
            except Exception as e:
                print(f"Warning: Failed to copy to Google Drive automatically: {str(e)}")
                
        print("\nImpRAG pipeline completed successfully!")

    # Synchronize processes before exit
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
