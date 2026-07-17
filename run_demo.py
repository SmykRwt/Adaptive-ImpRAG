import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from imprag.model import ImpRAGModel
from imprag.dataset import ImpRAGDataset, collate_fn, SyntheticTaskGenerator
from imprag.trainer import ImpRAGTrainer

def build_toy_dataset():
    # Toy knowledge corpus (passages containing facts)
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
    
    # Toy QA pairs with corresponding positive and negative passages
    qa_pairs = [
        {
            "query": "Q: What is the capital of France? A: ",
            "answer": "Paris",
            "pos": [corpus[0]],
            "negs": corpus[1:]
        },
        {
            "query": "Q: Who wrote Hamlet? A: ",
            "answer": "Shakespeare",
            "pos": [corpus[1]],
            "negs": [corpus[0]] + corpus[2:]
        },
        {
            "query": "Q: What is the largest ocean? A: ",
            "answer": "Pacific",
            "pos": [corpus[2]],
            "negs": corpus[:2] + corpus[3:]
        },
        {
            "query": "Q: What is the speed of light? A: ",
            "answer": "299792458 meters per second",
            "pos": [corpus[3]],
            "negs": corpus[:3] + corpus[4:]
        },
        {
            "query": "Q: What is the capital of Italy? A: ",
            "answer": "Rome",
            "pos": [corpus[4]],
            "negs": corpus[:4] + corpus[5:]
        },
        {
            "query": "Q: What is the longest river in Africa? A: ",
            "answer": "Nile",
            "pos": [corpus[5]],
            "negs": corpus[:5] + corpus[6:]
        },
        {
            "query": "Q: What force pulls objects together? A: ",
            "answer": "Gravity",
            "pos": [corpus[6]],
            "negs": corpus[:6] + corpus[7:]
        },
        {
            "query": "Q: Who developed relativity? A: ",
            "answer": "Einstein",
            "pos": [corpus[7]],
            "negs": corpus[:7] + corpus[8:]
        }
    ]
    
    # Let's add some synthetic phrase denoising and sentence generation tasks
    dataset_items = []
    for item in qa_pairs:
        dataset_items.append({
            "query": item["query"],
            "answer": item["answer"],
            "positive_passages": item["pos"],
            "negative_passages": item["negs"]
        })
        
    # Generate some synthetic tasks from our corpus
    for paragraph in corpus:
        # Phrase Denoising task
        pd_query, pd_ans = SyntheticTaskGenerator.generate_phrase_denoising(paragraph)
        if pd_ans:
            dataset_items.append({
                "query": pd_query,
                "answer": pd_ans,
                "positive_passages": [paragraph],
                "negative_passages": [p for p in corpus if p != paragraph]
            })
            
        # Sentence Generation task
        sg_query, sg_ans = SyntheticTaskGenerator.generate_sentence_gen(paragraph)
        if sg_ans:
            dataset_items.append({
                "query": sg_query,
                "answer": sg_ans,
                "positive_passages": [paragraph],
                "negative_passages": [p for p in corpus if p != paragraph]
            })
            
    return dataset_items

def main():
    print("=" * 60)
    print("ImpRAG Basic Implementation - Small-Scale Simulation")
    print("=" * 60)
    
    # 1. Setup tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print("Failed to load pretrained GPT2 tokenizer. Creating custom pad token configurations.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=False)
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Build tiny custom model to run quickly on CPU/GPU
    print("Initializing tiny GPT-2 model (4 layers, 128 hidden dim)...")
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
    
    # 3. Instantiate ImpRAG model wrapper
    # For a 4-layer model, we slice:
    # L_B = layers 0..1 (b=1)
    # L_M = layers 1..2 (t=2)
    # L_T = layers 3..3
    # Sharing layer b=1, reader layers = [1, 2] (cross-attention active)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ImpRAGModel(base_model, b=1, t=2, k_passages=2, max_passage_len=32)
    model.to(device)
    
    # 4. Prepare data loaders
    print("Building datasets...")
    data_items = build_toy_dataset()
    print(f"Generated dataset containing {len(data_items)} items.")
    
    # Split into train and eval (quick splits)
    train_data = data_items[:int(len(data_items)*0.8)]
    eval_data = data_items[int(len(data_items)*0.8):]
    
    train_dataset = ImpRAGDataset(train_data)
    eval_dataset = ImpRAGDataset(eval_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer, max_query_len=64, max_passage_len=32, num_candidates=5)
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, tokenizer, max_query_len=64, max_passage_len=32, num_candidates=5)
    )
    
    # 5. Initialize Optimizer and Trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = ImpRAGTrainer(
        model=model,
        optimizer=optimizer,
        warmup_epochs=2,     # 2 epochs NCE Warmup
        total_epochs=4,      # 4 epochs total (2 Warmup, 2 Distillation)
        lambda_ret=1.0,
        device=device
    )
    
    # 6. Evaluate BEFORE training
    print("\n" + "-" * 40)
    print("Evaluating Model Before Training...")
    init_em, init_recall = trainer.evaluate(eval_loader, tokenizer)
    print(f"Initial Exact Match (EM): {init_em * 100:.2f}%")
    print(f"Initial Retrieval Recall: {init_recall * 100:.2f}%")
    print("-" * 40 + "\n")
    
    # 7. Run training loop
    print("Starting ImpRAG Two-Stage Training (2 Warmup, 2 Distillation epochs)...")
    for epoch in range(4):
        avg_loss, avg_gen, avg_ret = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch+1}/4 complete: Avg Loss = {avg_loss:.4f} (Gen = {avg_gen:.4f}, Ret = {avg_ret:.4f})\n")
        
    # 8. Evaluate AFTER training
    print("\n" + "-" * 40)
    print("Evaluating Model After Training...")
    final_em, final_recall = trainer.evaluate(eval_loader, tokenizer)
    print(f"Final Exact Match (EM): {final_em * 100:.2f}%")
    print(f"Final Retrieval Recall: {final_recall * 100:.2f}%")
    print("-" * 40 + "\n")
    
    # 9. Demonstrate inference / generation
    print("Demonstrating Inference Sample:")
    sample = eval_data[0]
    print(f"Query: {sample['query']}")
    print(f"Expected Answer: {sample['answer']}")
    
    # Create candidate passages and compute scores to retrieve top passages
    # For a real run, we encode the full database, but here we use the sample's positive and negatives
    test_passages = sample["positive_passages"] + sample["negative_passages"][:3]
    
    test_passage_encoding = tokenizer(test_passages, padding=True, truncation=True, max_length=32, return_tensors="pt")
    test_passage_ids = test_passage_encoding.input_ids.to(device)
    test_passage_mask = test_passage_encoding.attention_mask.to(device)
    
    test_query_encoding = tokenizer([sample["query"]], return_tensors="pt")
    test_query_ids = test_query_encoding.input_ids.to(device)
    test_query_mask = test_query_encoding.attention_mask.to(device)
    
    # Get retriever scores
    with torch.no_grad():
        E_q = model.get_retriever_embeddings(test_query_ids, attention_mask=test_query_mask, is_query=True)
        E_p = model.get_retriever_embeddings(test_passage_ids, attention_mask=test_passage_mask, is_query=False)
        scores = torch.matmul(E_q, E_p.T).squeeze(0)
        
        # Select top-2 passages
        _, top_indices = torch.topk(scores, k=2)
        top_passages = [test_passages[idx.item()] for idx in top_indices]
        print(f"Retrieved Passages: {top_passages}")
        
        # Encode top-2 passages
        selected_passage_ids = test_passage_ids[top_indices]
        custom_cache = model.encode_passages(selected_passage_ids)
        
        # Generate answer
        gen_tokens = model.generate(test_query_ids, custom_cache, max_new_tokens=10)
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print(f"Generated Answer: {gen_text.strip()}")
        
    print("\nSimulation complete successfully!")

if __name__ == "__main__":
    main()
