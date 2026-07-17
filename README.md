# 🧠 Adaptive ImpRAG (Implicit Retrieval-Augmented Generation)

This repository contains a **paper-faithful baseline implementation** of the original **ImpRAG (Implicit Retrieval-Augmented Generation)** paper. It is designed to run efficiently on local hardware CPU/GPU resources (tested on standard 16GB RAM laptops) using a sliced **Qwen-2.5-1.5B** architecture.

---

## 📋 Architectural Overview

The codebase slices the 28-layer transformer model into three distinct functional groups, exactly matching the inference pipeline in Section 3.1 of the paper:

1. **Bottom Layers ($L_B$: Layers $0 \dots 14$)**: 
   * Acts as the **Retriever**.
   * Hooks into the `q_proj` and `k_proj` attention modules at layer $b=14$.
   * Applies **Grouped-Query Attention (GQA) group-averaging** to reduce query representations from 1536 to **256 dimensions** (matching the key projection space).
   * Applies **Mean-Pooling** over the active tokens to represent document and query semantics (preventing representation collapse on sentence-ending punctuation).
   * Uses **Dual-Centering** (subtracting document and query space means) to align the representations in the vector space for cosine similarity.
2. **Middle Layers ($L_M$: Layers $15 \dots 19$)**: 
   * Acts as the **Implicit Cache**.
   * Encodes the top-$k$ retrieved passages and prepends their keys/values into the attention states of layers $15 \dots 19$ using a `DynamicCache` injection during the generation pass.
3. **Top Layers ($L_T$: Layers $20 \dots 27$)**: 
   * Acts as the **Generator (Reader)**.
   * Autoregressively generates the final answer using the enriched cache.

---

## 🚀 Teammate Quickstart Guide

To run this project locally, your teammates should follow these steps:

### 1. Clone & Install Dependencies
First, clone the repository and install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Download the Model Checkpoint
Because the model weights folder is large (~3GB), it is excluded from GitHub. 
* Teammates must download the `imp_rag_checkpoint` folder (or download the pre-trained `Qwen/Qwen2.5-1.5B-Instruct` configuration) and place it in the project root directory.

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
