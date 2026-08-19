import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from imprag.model import ImpRAGModel
from imprag.dataset import ImpRAGDataset, collate_fn, SyntheticTaskGenerator
from imprag.trainer import ImpRAGTrainer

def build_toy_dataset():
    corpus = [
        "Paris is the capital city of France, known for the Eiffel Tower and museums.",
        "Shakespeare was an English playwright who wrote the tragedy Hamlet.",
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
        "Light travels at a speed of 299792458 meters per second in a vacuum.",
        "Rome is the historic capital of Italy, famous for the Colosseum.",
        "The Nile is the longest river in Africa, flowing northwards into the Mediterranean.",
        "Gravity is a fundamental force that attracts objects with mass toward each other.",
        "Albert Einstein developed the theory of relativity, a pillar of modern physics.",
        "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        "Mount Everest is the highest mountain on Earth, located in the Himalayas."
    ]
    
    qa_pairs = [
        {"query": "Q: What is the capital of France? A: ", "answer": "Paris", "pos": [corpus[0]], "negs": corpus[1:5]},
        {"query": "Q: Who wrote Hamlet? A: ", "answer": "Shakespeare", "pos": [corpus[1]], "negs": [corpus[0]] + corpus[2:5]},
        {"query": "Q: What is the largest ocean? A: ", "answer": "Pacific", "pos": [corpus[2]], "negs": corpus[:2] + corpus[3:5]},
        {"query": "Q: What is the speed of light? A: ", "answer": "299792458 meters per second", "pos": [corpus[3]], "negs": corpus[:3] + corpus[4:6]},
        {"query": "Q: What is the capital of Italy? A: ", "answer": "Rome", "pos": [corpus[4]], "negs": corpus[:4] + corpus[5:7]},
        {"query": "Q: What is the longest river in Africa? A: ", "answer": "Nile", "pos": [corpus[5]], "negs": corpus[:5] + corpus[6:8]},
        {"query": "Q: What force pulls objects together? A: ", "answer": "Gravity", "pos": [corpus[6]], "negs": corpus[:6] + corpus[7:9]},
        {"query": "Q: Who developed relativity? A: ", "answer": "Einstein", "pos": [corpus[7]], "negs": corpus[:7] + corpus[8:10]}
    ]
    
    dataset_items = []
    for item in qa_pairs:
        dataset_items.append({
            "query": item["query"],
            "answer": item["answer"],
            "positive_passages": item["pos"],
            "negative_passages": item["negs"]
        })
        
    for paragraph in corpus:
        pd_query, pd_ans = SyntheticTaskGenerator.generate_phrase_denoising(paragraph)
        if pd_ans:
            dataset_items.append({
                "query": pd_query,
                "answer": pd_ans,
                "positive_passages": [paragraph],
                "negative_passages": [p for p in corpus if p != paragraph][:4]
            })
            
        sg_query, sg_ans = SyntheticTaskGenerator.generate_sentence_gen(paragraph)
        if sg_ans:
            dataset_items.append({
                "query": sg_query,
                "answer": sg_ans,
                "positive_passages": [paragraph],
                "negative_passages": [p for p in corpus if p != paragraph][:4]
            })
            
    return dataset_items, corpus

def main():
    print("=" * 60)
    print("ImpRAG Baseline Simulation - Paper Faithful Implementation")
    print("=" * 60)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Initializing tiny GPT-2 model (4 layers, 128 hidden dim, 4 heads)...")
    config = GPT2Config(
        n_layer=4,
        n_embd=128,
        n_head=4,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    base_model = GPT2LMHeadModel(config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 4-layer model slicing: b=1 (layers 0..1), LM=1..2 (layers 1..2), LT=3..3
    model = ImpRAGModel(base_model, b=1, t=2, k_passages=2, max_passage_len=32, pooling_type="last_token")
    model.to(device)
    
    print("Building datasets...")
    data_items, corpus = build_toy_dataset()
    print(f"Generated dataset containing {len(data_items)} items.")
    
    train_data = data_items[:int(len(data_items)*0.8)]
    eval_data = data_items[int(len(data_items)*0.8):]
    
    train_dataset = ImpRAGDataset(train_data)
    eval_dataset = ImpRAGDataset(eval_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer, max_query_len=32, max_passage_len=32, max_positives_per_query=2, max_negatives_per_query=2)
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, tokenizer, max_query_len=32, max_passage_len=32, max_positives_per_query=2, max_negatives_per_query=2)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ImpRAGTrainer(
        model=model,
        optimizer=optimizer,
        warmup_epochs=2,     # 2 epochs NCE Warmup
        total_epochs=4,      # 4 epochs total (2 Warmup, 2 Distillation)
        lambda_ret=1.0,
        device=device
    )
    
    print("\n" + "-" * 40)
    print("Evaluating Model Before Training...")
    init_em, init_recall = trainer.evaluate(eval_loader, tokenizer, k_top=2)
    print(f"Initial Exact Match (EM): {init_em * 100:.2f}%")
    print(f"Initial Retrieval Recall: {init_recall * 100:.2f}%")
    print("-" * 40 + "\n")
    
    print("Starting ImpRAG Two-Stage Training (2 Warmup NCE, 2 Distillation KL epochs)...")
    for epoch in range(4):
        avg_loss, avg_gen, avg_ret = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch+1}/4 complete: Avg Loss = {avg_loss:.4f} (Gen = {avg_gen:.4f}, Ret = {avg_ret:.4f})\n")
        
    print("-" * 40)
    print("Evaluating Model After Training...")
    final_em, final_recall = trainer.evaluate(eval_loader, tokenizer, k_top=2)
    print(f"Final Exact Match (EM): {final_em * 100:.2f}%")
    print(f"Final Retrieval Recall: {final_recall * 100:.2f}%")
    print("-" * 40 + "\n")
    
    print("Demonstrating Inference Sample with Full Attention Concatenated Passages:")
    sample = eval_data[0]
    print(f"Query: {sample['query']}")
    print(f"Expected Answer: {sample['answer']}")
    
    test_passages = sample["positive_passages"] + sample["negative_passages"][:3]
    test_passage_enc = tokenizer(test_passages, padding=True, truncation=True, max_length=32, return_tensors="pt")
    test_p_ids = test_passage_enc.input_ids.to(device)
    test_p_mask = test_passage_enc.attention_mask.to(device)
    
    test_q_enc = tokenizer([sample["query"]], return_tensors="pt")
    test_q_ids = test_q_enc.input_ids.to(device)
    test_q_mask = test_q_enc.attention_mask.to(device)
    
    with torch.no_grad():
        E_q = model.get_retriever_embeddings(test_q_ids, attention_mask=test_q_mask, is_query=True)
        E_p = model.get_retriever_embeddings(test_p_ids, attention_mask=test_p_mask, is_query=False)
        scores = torch.matmul(E_q, E_p.T).squeeze(0)
        
        _, top_indices = torch.topk(scores, k=min(2, len(test_passages)))
        retrieved_passages = [test_passages[idx.item()] for idx in top_indices]
        print(f"Retrieved Passages: {retrieved_passages}")
        
        selected_p_ids = test_p_ids[top_indices].unsqueeze(0)
        custom_cache = model.encode_passages(selected_p_ids, k_passages=len(top_indices), encoding_mode="concatenated")
        
        gen_tokens = model.generate(test_q_ids, custom_cache, max_new_tokens=10)
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print(f"Generated Output: {gen_text.strip()}")
        
    print("\nBaseline simulation completed successfully!")

if __name__ == "__main__":
    main()
