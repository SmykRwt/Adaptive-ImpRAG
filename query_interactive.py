import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from imprag.model import ImpRAGModel
from imprag.retriever import ImpRAGFAISSIndex

def main():
    print("=" * 60)
    print("ImpRAG Interactive Query Interface")
    print("=" * 60)
    
    checkpoint_dir = "imp_rag_checkpoint"
    index_path = "imp_rag_wiki.index"
    corpus_path = "wiki_passages.json"
    
    # 1. Verification
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint folder '{checkpoint_dir}' not found.")
        print("Please download it from Google Drive and place it here.")
        return
        
    if not os.path.exists(index_path):
        print(f"Error: FAISS index file '{index_path}' not found.")
        print("Please download it from Google Drive and place it here.")
        return
        
    if not os.path.exists(corpus_path):
        print(f"Error: Wikipedia corpus file '{corpus_path}' not found.")
        print("Please make sure 'wiki_passages.json' is here.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Load resources
    print("Loading Wikipedia passages...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        passages = json.load(f)
    print(f"Loaded {len(passages)} passages.")
    
    print("Loading FAISS retrieval index...")
    faiss_index = ImpRAGFAISSIndex.load(index_path)
    print(f"Loaded index containing {faiss_index.get_num_vectors()} passage vectors.")
    
    print("Loading trained model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
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
        
    model = ImpRAGModel(base_model, b=b, t=t, k_passages=2, max_passage_len=64)
    model.to(device)
    model.eval()
    
    print("\nInitialization Complete! Ask any question.")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Exiting interactive interface. Goodbye!")
                break
                
            # Format prompt for the model
            formatted_query = f"Q: {query} A: "
            
            # Embed the query
            q_enc = tokenizer([formatted_query], return_tensors="pt")
            q_ids = q_enc.input_ids.to(device)
            q_mask = q_enc.attention_mask.to(device)
            
            with torch.no_grad():
                E_q = model.get_retriever_embeddings(q_ids, attention_mask=q_mask, is_query=True)
                E_q_norm = F.normalize(E_q, p=2, dim=-1)
                
                # Query FAISS index for top 5 documents
                distances, indices = faiss_index.search(E_q_norm.float().cpu().numpy(), k=5)
                ret_passages = [passages[idx] for idx in indices[0] if idx != -1]
                
                print("\n" + "-" * 30 + " RETRIEVED CONTEXTS " + "-" * 30)
                for rank, text in enumerate(ret_passages[:2], 1):
                    print(f"[{rank}] {text}")
                print("-" * 80)
                
                # Retrieve top-2 passages and encode to KV states for reader
                retrieved_passage_ids = tokenizer(
                    ret_passages[:2], 
                    padding=True, 
                    truncation=True, 
                    max_length=64, 
                    return_tensors="pt"
                ).input_ids.to(device)
                
                custom_cache = model.encode_passages(retrieved_passage_ids)
                
                # Generate Answer
                print("Generating answer...")
                gen_tokens = model.generate(q_ids, custom_cache, max_new_tokens=25)
                
                # Decode output directly (gen_tokens only contains new tokens)
                gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
                
                print(f"\n>> Generated Answer: {gen_text}")
                print("=" * 80)
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
