import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from imprag.model import ImpRAGModel
from imprag.retriever import ImpRAGFAISSIndex
import gradio as gr

# Global variables to load models once
tokenizer = None
model = None
faiss_index = None
passages = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_resources():
    global tokenizer, model, faiss_index, passages
    checkpoint_dir = "imp_rag_checkpoint"
    index_path = "imp_rag_wiki.index"
    corpus_path = "wiki_passages.json"
    
    if not os.path.exists(checkpoint_dir) or not os.path.exists(index_path) or not os.path.exists(corpus_path):
        return False, "Error: Trained files missing. Please download imp_rag_checkpoint, imp_rag_wiki.index, and wiki_passages.json into this folder."
        
    try:
        # Load passages
        with open(corpus_path, "r", encoding="utf-8") as f:
            passages = json.load(f)
            
        # Load index
        faiss_index = ImpRAGFAISSIndex.load(index_path)
        
        # Load model & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Use bfloat16 for GPU, float16 for CPU to prevent Windows 32GB RAM OOM crashes
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        
        base_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        )
        
        # Slice and wrap the model dynamically based on number of layers in checkpoint
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
            
        model = ImpRAGModel(base_model, b=b, t=t, k_passages=5, max_passage_len=128)
        model.to(device)
        model.eval()
        return True, "All models loaded successfully!"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Failed to load resources: {str(e)}"

# Attempt to load immediately when launching
ready, load_msg = load_resources()

def qa_interface(query):
    if not ready:
        return "Model not loaded. Check if checkpoint folder is in the project root.", "No context retrieved."
        
    if not query.strip():
        return "Please enter a valid question.", ""
        
    try:
        formatted_query = f"Q: {query} A: "
        q_enc = tokenizer([formatted_query], return_tensors="pt")
        q_ids = q_enc.input_ids.to(device)
        q_mask = q_enc.attention_mask.to(device)
        
        with torch.no_grad():
            # 1. Embed query
            E_q = model.get_retriever_embeddings(q_ids, attention_mask=q_mask, is_query=True)
            E_q_norm = F.normalize(E_q, p=2, dim=-1)
            
            # 2. Search FAISS index
            distances, indices = faiss_index.search(E_q_norm.float().cpu().numpy(), k=5)
            ret_passages = [passages[idx] for idx in indices[0] if idx != -1]
            
            # Ensure we always have at least k_passages (5) to prevent batch size shape crashes
            while len(ret_passages) < model.k_passages:
                ret_passages.append("No context available.")
                
            # Format context display
            context_markdown = "### 📚 Top-5 Retrieved Wikipedia Contexts:\n\n"
            for rank, text in enumerate(ret_passages[:5], 1):
                context_markdown += f"**Passage {rank}**\n> {text}\n\n"
                
            # 3. Retrieve top-k passages and encode to KV states for reader
            retrieved_passage_ids = tokenizer(
                ret_passages[:model.k_passages], 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).input_ids.to(device)
            
            custom_cache = model.encode_passages(retrieved_passage_ids)
            
            # 4. Generate Answer
            gen_tokens = model.generate(q_ids, custom_cache, max_new_tokens=30)
            
            # Decode output directly
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
            
            return gen_text, context_markdown
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing query: {str(e)}", f"Error during retrieval: {str(e)}"

# Build modern Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🧠 Adaptive ImpRAG (Implicit Retrieval-Augmented Generation)
        ### Capstone Project Baseline Demo — Meta-Llama-3-8B-Instruct
        Ask a question to search the Simple Wikipedia corpus and generate an answer using implicit middle-layer caching.
        """
    )
    
    # Show loading status
    status_color = "green" if ready else "red"
    gr.Markdown(f"**System Status**: <span style='color:{status_color};'>{load_msg}</span>")
    
    with gr.Row():
        with gr.Column(scale=5):
            query_input = gr.Textbox(
                label="Enter your Question:", 
                placeholder="e.g., Who wrote the play Hamlet?",
                lines=2
            )
            submit_btn = gr.Button("Query Model", variant="primary")
            answer_output = gr.Textbox(
                label="Generated Answer:", 
                placeholder="Waiting for query...",
                interactive=False,
                lines=4
            )
            
        with gr.Column(scale=5):
            context_output = gr.Markdown(value="*Retrieved Wikipedia contexts will appear here...*")
            
    # Bind actions
    submit_btn.click(
        fn=qa_interface,
        inputs=query_input,
        outputs=[answer_output, context_output]
    )
    
    query_input.submit(
        fn=qa_interface,
        inputs=query_input,
        outputs=[answer_output, context_output]
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        prevent_thread_lock=False
    )
