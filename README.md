# 🧠 Adaptive ImpRAG (Implicit Retrieval-Augmented Generation)

This repository contains a **paper-faithful baseline implementation** of the original **ImpRAG (Implicit Retrieval-Augmented Generation)** paper. It is designed to run efficiently on GPU resources using a sliced **Llama-3-3B** (`meta-llama/Llama-3.2-3B-Instruct`) architecture with dynamic layer slicing for larger/smaller models.

---

## 📋 Architectural & Training Overview

The codebase slices the transformer model into three distinct functional groups, matching the inference pipeline in Section 3.1 of the paper:

1. **Bottom Layers ($L_B$: Layers $0 \dots 14$)**: 
   * Acts as the **Retriever**.
   * Hooks into the `q_proj` and `k_proj` attention modules at layer $b=14$ (dynamically shifts to $b=15$ for 32-layer Llama-3-8B).
   * Applies **Grouped-Query Attention (GQA) group-averaging** to reduce query representations to **256 dimensions**.
   * Uses **Dual-Centering** to align representations.
2. **Middle Layers ($L_M$: Layers $15 \dots 19$)**: 
   * Acts as the **Implicit Cache** (dynamically shifts to $15 \dots 23$ for Llama-3-8B).
   * Encodes the top-$k$ retrieved passages and prepends their keys/values into the attention states of middle layers during generation.
3. **Top Layers ($L_T$: Layers $20 \dots 27$)**: 
   * Acts as the **Generator (Reader)**.
   * Autoregressively generates the final answer using the enriched cache.

### 🔄 In-Batch Negatives Training Dynamics
The training loop utilizes **in-batch negatives** (exactly resembling the paper's execution details):
* **InfoNCE Warmup**: Computes dot-product cosine similarity scores between all queries and positive passages in a batch, training the retriever via multi-class cross-entropy (where the diagonal element is the positive target).
* **Self-Distillation (KL)**: Replicates the query caches over in-batch candidates to compute teacher generation perplexities, distilling the resulting soft target distribution directly to the retriever.

---

## 🚀 Teammate Quickstart Guide

To run this project locally, your teammates should follow these steps:

### 1. Clone & Install Dependencies
First, clone the repository and install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Download the Model Checkpoint
* Teammates should download the `imp_rag_checkpoint` folder (or log in via `huggingface-cli login` to fetch `meta-llama/Llama-3.2-3B-Instruct`) and place it in the project root directory.

### 3. Place/Rebuild the FAISS Index
* Ensure the 3 pre-built mean-pooled index files are in the root directory:
  - `imp_rag_wiki.index` (The FAISS retrieval vector database)
  - `imp_rag_wiki.index.mean.npy` (Document mean vector for centering)
  - `imp_rag_wiki.index.query_mean.npy` (Query mean vector for centering)
* *Note: If they need to rebuild the index from scratch on a GPU (e.g. Google Colab), they can run `python rebuild_index.py` which takes under 30 seconds on a GPU.*

### 4. Launch the Web Interface (Gradio)
Start the Gradio Web App locally:
```bash
python app_web.py
```
Open the local URL in your web browser: **`http://127.0.0.1:7860`**

---

## 🛠️ Diagnostics & Evaluation Scripts

To test the health and mathematics of the retrieval pipeline, we included two diagnostic scripts:

### 1. Retrieval Debugger
Run this script to inspect query representation norms, raw corpus checks, and top-10 retrieval similarity rankings for any query:
```bash
python debug_retrieval.py "Who was Cicely Mary Barker?"
```

### 2. Retrieval Metrics Evaluator
Run this script to calculate standard retrieval performance metrics (**Recall@1**, **Recall@5**, **Recall@10**, and **MRR**) on the evaluation dataset:
```bash
python evaluate_retriever.py
```
