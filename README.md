# 🧠 Adaptive ImpRAG (Implicit Retrieval-Augmented Generation)

This repository contains a **paper-faithful baseline implementation** of the original **ImpRAG (Implicit Retrieval-Augmented Generation)** paper. It is fully scaled to run on high-performance single-GPU (e.g., NVIDIA A100 40GB/80GB) or multi-GPU cluster configurations, using a sliced **Llama-3-8B** (`meta-llama/Meta-Llama-3-8B-Instruct`) architecture by default, with dynamic support for $k=5$ injected passages and 128-token passage budgets.

---

## 📋 Architectural Overview

The codebase slices the base transformer model into three distinct functional groups, matching the inference pipeline in Section 3.1 of the paper:

1. **Bottom Layers ($L_B$: Layers $0 \dots b$)**: 
   * Acts as the **Retriever**.
   * Hooks into the `q_proj` and `k_proj` attention modules at layer $b$ (automatically set: $b=14$ for 28-layer models like Llama-3-3B; $b=15$ for 32-layer models like Llama-3-8B).
   * Applies **Grouped-Query Attention (GQA) group-averaging** to reduce query representation dimensions to **256**.
   * Uses **Dual-Centering** to align representations and combat anisotropy.
2. **Middle Layers ($L_M$: Layers $b \dots t$)**: 
   * Acts as the **Implicit Cache** (automatically set: layers $14 \dots 19$ for 3B; layers $15 \dots 23$ for 8B).
   * Encodes the top-$k$ retrieved passages and prepends their keys/values into the attention states of middle layers during generation.
3. **Top Layers ($L_T$: Layers $t \dots \text{end}$)**: 
   * Acts as the **Generator (Reader)** (automatically set: layers $20 \dots 27$ for 3B; layers $24 \dots 31$ for 8B).
   * Autoregressively generates the final answer using the enriched cache.

---

## 🔄 Training & Optimization Dynamics

The training loop implements a two-stage training process (InfoNCE Warmup + KL Distillation) with high-fidelity performance upgrades matching the paper's cluster execution details:

### 1. In-Batch Negatives InfoNCE
During the **Warmup Stage** (Epochs 1..warmup_epochs), query embeddings $E_q \in \mathbb{R}^{B \times D}$ and positive passage embeddings $E_p \in \mathbb{R}^{B \times D}$ in the batch are dot-product matched:
$$\text{scores} = \frac{E_q E_p^T}{0.05} \quad (\text{shape } [B, B])$$
The diagonal elements act as positive matches, while all off-diagonal items in the batch act as negatives, trained using standard cross-entropy.

### 2. In-Batch Self-Distillation (KL)
During the **Self-Distillation Stage**, the teacher model evaluates target response log-likelihoods conditioned on every positive passage in the batch, yielding a soft target distribution $L \in \mathbb{R}^{B \times B}$. The retriever is trained by minimizing the KL-divergence between the retriever score distributions and these soft targets.

### 3. Selective Weight Freezing
To prevent catastrophic forgetting of the model's pre-trained generation capabilities, **all reader and generator weights are frozen**. Only the key ($W_K$) and query ($W_Q$) projection matrices of the bottom layers group ($L_B$, layers $0 \dots b$) accumulate gradients and update during optimization, speeding up training by 3x.

### 4. Distributed Multi-GPU (DDP) Scaling
The codebase supports PyTorch **Distributed Data Parallel (DDP)**:
* Uses `DistributedSampler` to split dataset batches cleanly across GPUs.
* Uses **embedding all-gather** (`dist.all_gather`) to collect embeddings across all active GPUs, expanding the in-batch negatives pool from $B-1$ to $(N \times B) - 1$.
* Restricts index building, centering vector calculations, and final metric evaluations to **Rank 0 (Main GPU)** to prevent file-write collisions.

---

## 🚀 Teammate Quickstart Guide

To run this project locally or on a GPU server, follow these steps:

### 1. Clone & Install Dependencies
First, clone the repository and install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Gain Llama-3 Hugging Face Access
Ensure you have logged in via the Hugging Face CLI to authenticate download permissions for the gated `meta-llama` models:
```bash
huggingface-cli login
```

### 3. Prepare the Wikipedia Corpus (Simple Wikipedia, ~2,000,000 passages)
Prepare the evaluation corpus. Running this downloads Simple Wikipedia and chunks it into 120-word passages:
```bash
python prepare_wiki_corpus.py --dataset wikipedia_simple
```

### 4. Mine Pseudo-Labels (Streams 50,000 training queries)
Stream NQ and HotpotQA training queries, batch-embed them on the GPU, and use GPU-accelerated FAISS search to mine pseudo-labels:
```bash
python generate_pseudo_labels.py --max_queries 25000
```

### 5. Launch Training
#### Option A: Single-GPU training (with Gradient Accumulation & AMP)
```bash
python train_pipeline.py --epochs 6 --accumulation_steps 16 --use_amp
```
#### Option B: Distributed Multi-GPU DDP training (e.g., 4 GPUs)
```bash
torchrun --nproc_per_node=4 train_pipeline.py --epochs 6 --accumulation_steps 8 --use_amp
```

### 6. Launch the Web Interface (Gradio)
To visually search passages and chat with the model in real-time, run:
```bash
python app_web.py
```
Open **`http://127.0.0.1:7860`** in your browser.

---

## 🛠️ Diagnostics & Evaluation Scripts

To test the health and mathematics of the retrieval pipeline, we included two diagnostic scripts:

### 1. Retrieval Debugger
Inspect query representation norms, raw corpus keyword scans, and top-10 retrieval similarity rankings for any query:
```bash
python debug_retrieval.py "Who was Cicely Mary Barker?"
```

### 2. Retrieval Metrics Evaluator
Calculate retrieval performance metrics (**Recall@1**, **Recall@5**, **Recall@10**, and **MRR**) on the evaluation dataset:
```bash
python evaluate_retriever.py
```
