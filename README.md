# 🧠 Adaptive ImpRAG (Implicit Retrieval-Augmented Generation)

This repository contains:
1. A **100% paper-faithful baseline implementation** of the original **ImpRAG (Implicit Retrieval-Augmented Generation)** paper (*Zhang et al., Meta / Rutgers*).
2. The complete **ADAPTIVE ImpRAG** architecture introducing dynamic adaptivity across **4 key dimensions**:
   - **Dimension 1: Dynamic Retrieval Decision (When to Retrieve)** — Bypasses FAISS search and KV injection for parametric queries.
   - **Dimension 2: Dynamic $k$ Allocation (How Much to Retrieve)** — Adaptively allocates $k \in \{1, 2, 5, 10\}$ based on score distribution entropy and top-margin.
   - **Dimension 3: Adaptive Layer Slicing & Injection Boundaries ($b(q), t(q)$)** — Dynamically routes cache injection depth by task complexity.
   - **Dimension 4: Adaptive GQA Head Attention Pooling** — Dynamic query-conditioned learned head weighting replacing uniform head averaging.

---

## 📋 Architectural Overview

### 1. Paper-Faithful ImpRAG Slicing
An $N$-layer decoder-only language model (e.g. `Meta-Llama-3-8B-Instruct`) is partitioned vertically into three groups:

1. **Bottom Layers ($L_B$: Layers $0 \dots b$, default $b=7$)**:
   - Acts as the **Retriever**.
   - Hooks into $W_Q$ and $W_K$ projection modules at layer $b$.
   - Applies Grouped-Query Attention (GQA) group-averaging ($g = h_q / h_k$) to query projections.
   - Pools last token to extract query embedding $E_q \in \mathbb{R}^{h_k d_h}$ and passage embedding $E_p \in \mathbb{R}^{h_k d_h}$.
   - Inner product similarity: $s(q, p) = E_q \cdot E_p$.
   - Dual-Centering applied during FAISS retrieval to combat representation anisotropy.

2. **Middle Layers ($L_M$: Layers $b \dots t$, default $t=23$ for 8B; $t=19$ for 3B)**:
   - Acts as the **Implicit Cache / Reader**.
   - Uses **Full Attention Concatenated Passage Encoding** (Section 3.1 & Appendix A, Table 6) where all $k$ retrieved passages are concatenated and jointly encoded.
   - Injects passage KV states into `DynamicCache` strictly inside layers $b \dots t$.

3. **Top Layers ($L_T$: Layers $t+1 \dots N-1$)**:
   - Acts as the **Generator**.
   - Cross-attention to passage KV cache is disabled in top layers to reduce memory overhead.
   - Autoregressively generates answers with shifted position IDs ($k \cdot l_{\max}$).

---

## 🌟 Adaptive ImpRAG (The 4 Dimensions)

```
                       Input Query q
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Bottom Layers L_B (0 .. b)  │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   Adaptive GQA Head Pooling  │ ──► Dynamic Learned Head Weights
              │    (Dimension 4: α_h(q))     │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    Dynamic Retrieval Gate    │
              │  (Dimension 1: When to Ret)  │
              └──────┬────────────────┬──────┘
                     │                │
          Parametric │                │ Low-Confidence /
        Bypass (k=0) │                │ Multi-Hop (k > 0)
                     │                ▼
                     │ ┌──────────────────────────────┐
                     │ │      FAISS Index Search      │
                     │ └──────────────┬───────────────┘
                     │                │ Top Candidate Scores
                     │                ▼
                     │ ┌──────────────────────────────┐
                     │ │  Dynamic k Budget Allocator  │ ──► k ∈ {1, 2, 5, 10}
                     │ │ (Dimension 2: Entropy/Margin)│
                     │ └──────────────┬───────────────┘
                     │                │
                     │                ▼
                     │ ┌──────────────────────────────┐
                     │ │  Adaptive Boundary Router    │ ──► [b(q), t(q)] Depth
                     │ │  (Dimension 3: Cache Slicing)│
                     │ └──────────────┬───────────────┘
                     │                │ Concatenated Passage KV
                     │                ▼
                     │ ┌──────────────────────────────┐
                     │ │  Middle Layers L_M (b .. t)  │◄── Dynamic Cache Injection
                     │ └──────────────┬───────────────┘
                     │                │
                     └───────────────►▼
                       ┌──────────────────────────────┐
                       │  Top Layers L_T (t+1 .. N-1) │
                       │    Autoregressive Decoder    │
                       └──────────────────────────────┘
```

---

## 🔄 Two-Stage Multi-Task Training

The training pipeline optimizes the joint objective (Eq 2):
$$\mathcal{J} = \mathcal{J}_{	ext{gen}}(r \mid q, \mathcal{C}) + \lambda \cdot \mathcal{J}_{	ext{ret}}(q, \mathcal{C})$$

1. **Warmup Stage (Epochs 1..warmup_epochs)**:
   - Multi-Label NCE loss (Eq 3) over pseudo-positives $\mathcal{P}(q)$ and hard negatives $\mathcal{N}_h(q)$ with in-batch negatives.
2. **Self-Distillation Stage (Remaining epochs)**:
   - KL-divergence distillation (Eq 4-6) between teacher LM response likelihoods $P_T(p \mid q, r)$ and retriever distribution $P_R(p \mid q)$.
3. **Weight Freezing**:
   - Generator/reader layers frozen; updates applied to $W_Q, W_K$ of layers $0 \dots b$ and adaptive routing modules.

---

## 🚀 Quickstart & Verification

### 1. Run Verification Test Suites
Verify Baseline ImpRAG and all 4 dimensions of Adaptive ImpRAG:
```bash
python test_baseline_verification.py
python test_adaptive_verification.py
```

### 2. Run Small-Scale Demo Simulation
```bash
python run_demo.py
```

### 3. Launch Interactive Gradio Web Interface
```bash
python app_web.py
```
Open **`http://127.0.0.1:7860`** to interact with the full 4D Adaptive system and view real-time decision telemetry.

### 4. Full GPU / DDP Training Pipeline
```bash
python train_pipeline.py --epochs 6 --accumulation_steps 8 --use_amp
```
