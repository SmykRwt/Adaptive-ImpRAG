# 🧠 Adaptive ImpRAG (Implicit Retrieval-Augmented Generation)

This repository contains:
1. A **100% paper-faithful baseline implementation** of the original **ImpRAG (Implicit Retrieval-Augmented Generation)** paper (*Zhang et al., Meta / Rutgers*).
2. The complete **ADAPTIVE ImpRAG** architecture introducing dynamic adaptivity across **4 key dimensions**:
   - **Dimension 1: Dynamic Retrieval Decision (When to Retrieve)** — Bypasses FAISS search and KV injection for parametric queries.
   - **Dimension 2: Dynamic $k$ Allocation (How Much to Retrieve)** — Adaptively allocates $k \in \{1, 2, 5, 10\}$ based on score distribution entropy and top-margin.
   - **Dimension 3: Adaptive Layer Slicing & Injection Boundaries ($b(q), t(q)$)** — Dynamically routes cache injection depth by task complexity.
   - **Dimension 4: Grouped Query Head Pooling for Retrieval Embedding** — Replaces uniform mean pooling of grouped query-head projections with query-conditioned learned weighted pooling to construct $E_q$.

---

## 📋 Architectural Overview

### 1. Paper-Faithful ImpRAG Slicing
An $N$-layer decoder-only language model (e.g. `Meta-Llama-3-8B-Instruct`) is partitioned vertically into three groups:

1. **Bottom Layers ($L_B$: Layers $0 \dots b$, default $b=7$)**:
   - Acts as the **Retriever**.
   - Hooks into $W_Q$ and $W_K$ projection modules at boundary layer $b$.
   - At layer $b$, the **last token's** query projection is extracted (the last token aggregates full sequence context due to causal attention).
   - The 32 query-head projections at layer $b$ are organized using the **GQA group structure** ($g = h_q / h_k = 32 / 8 = 4$ query heads per KV group). 
   - > **Note**: In standard Llama-3 GQA attention, query heads are **not** averaged — each $Q_1, Q_2, Q_3, Q_4$ computes its own attention against the shared $K, V$ independently. The grouped pooling described below is a **retrieval-specific step** applied separately from the model's internal attention computation.
   - **Baseline ImpRAG** applies uniform mean pooling across the 4 query heads in each group to construct a compact retrieval embedding:
     $$E_{q,h} = \frac{q_{h,1} + q_{h,2} + q_{h,3} + q_{h,4}}{4}$$
   - This yields $E_q \in \mathbb{R}^{h_k \cdot d_h}$ (e.g., $8 \times 128 = 1024$ dimensions for Llama-3-8B).
   - Passage embeddings $E_p$ are similarly extracted using $W_K$ projections from passage tokens.
   - Inner product similarity: $s(q, p) = E_q \cdot E_p$.
   - Dual-Centering applied during FAISS retrieval to combat representation anisotropy:
     $$\tilde{E}_q = E_q - \mu_q, \quad \tilde{E}_p = E_p - \mu_p$$

2. **Middle Layers ($L_M$: Layers $b \dots t$, default $t=23$ for 8B; $t=19$ for 3B)**:
   - Acts as the **Implicit Cache / Reader**.
   - Uses **Full Attention Concatenated Passage Encoding** (Section 3.1 & Appendix A, Table 6) where all $k$ retrieved passages are concatenated and jointly encoded.
   - Injects passage KV states into `DynamicCache` strictly inside layers $b \dots t$.
   - Query token position IDs are shifted by passage length: `position_ids = arange(L_pass, L_pass + L_query)`.

3. **Top Layers ($L_T$: Layers $t+1 \dots N-1$)**:
   - Acts as the **Generator**.
   - Cross-attention to passage KV cache is disabled in top layers to reduce memory overhead.
   - Autoregressively generates answers conditioned on the enriched middle-layer hidden states.

---

## 🌟 Adaptive ImpRAG (The 4 Dimensions)

```
                       Input Query q
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Bottom Layers L_B (0 .. b)  │
              └──────────────┬───────────────┘
                             │ Last token query projections
                             │ [batch, 32 Q-heads, 128]
                             ▼
              ┌──────────────────────────────────────────────┐
              │  Grouped Query Head Pooling for Retrieval    │ ──► Dimension 4
              │  (Adaptive weighted pooling, not Llama GQA  │     α_h(q) learned weights
              │   attention — a retrieval-specific step)     │     replaces uniform mean
              └──────────────┬───────────────────────────────┘
                             │ E_q ∈ R^1024
                             ▼
              ┌──────────────────────────────┐
              │    Dynamic Retrieval Gate    │ ──► Dimension 1: Should we search?
              │  (Dimension 1: When to Ret)  │     p(retrieve | E_q) via gate_net
              └──────┬───────────────┬───────┘
                     │               │
         Parametric  │               │ Low-Confidence /
        Bypass (k=0) │               │ Multi-Hop (k > 0)
                     │               ▼
                     │ ┌──────────────────────────────┐
                     │ │      FAISS Index Search      │
                     │ └──────────────┬───────────────┘
                     │                │ Top Candidate Scores
                     │                ▼
                     │ ┌──────────────────────────────┐
                     │ │  Dynamic k Budget Allocator  │ ──► Dimension 2: k ∈ {1, 2, 5, 10}
                     │ │ (Dimension 2: Entropy/Margin)│     based on score entropy & margin
                     │ └──────────────┬───────────────┘
                     │                │
                     │                ▼
                     │ ┌──────────────────────────────┐
                     │ │  Adaptive Boundary Router    │ ──► Dimension 3: Layer Depth [b(q), t(q)]
                     │ │  (Dimension 3: Cache Slicing)│     Tier 1: [4,14] Tier 2: [7,20] Tier 3: [7,26]
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

### Dimension 4 — Grouped Query Head Pooling for Retrieval Embedding

> **Important Distinction**: Standard Llama-3 GQA attention does **not** average query heads. In transformer self-attention, each of the 4 query heads in a group independently computes `Q_i · K^T` against the shared KV head. This is purely an efficiency mechanism.
>
> Our **retrieval-specific pooling step** is separate and operates after the self-attention block completes, using the GQA grouping structure to organize query-head projections before combining them into a compact $E_q$ vector for FAISS search.

**Baseline ImpRAG (Uniform Mean)**:
$$E_{q,h} = \frac{1}{g} \sum_{i=1}^{g} q_{h,i} \quad \text{(equal weight = } \frac{1}{4} \text{ per head)}$$

**Adaptive ImpRAG (Learned Weighted Pooling)**:
$$\alpha_{h,i}(q) = \text{Softmax}\!\left(\frac{\langle q_{h,i},\, W_h \rangle}{\sqrt{d_h}}\right), \quad E_{q,h} = \sum_{i=1}^{g} \alpha_{h,i}(q)\, q_{h,i}$$

This allows the retriever to **dynamically suppress noisy heads** and **amplify informative heads** based on the input query.

---

## 🔄 Two-Stage Multi-Task Training

The training pipeline optimizes the joint objective (Eq 2):
$$\mathcal{J} = \mathcal{J}_{\text{gen}}(r \mid q, \mathcal{C}) + \lambda \cdot \mathcal{J}_{\text{ret}}(q, \mathcal{C})$$

1. **Warmup Stage (Epochs 1..warmup_epochs)**:
   - Multi-Label NCE loss (Eq 3) over pseudo-positives $\mathcal{P}(q)$ and hard negatives $\mathcal{N}_h(q)$ with in-batch negatives.
2. **Self-Distillation Stage (Remaining epochs)**:
   - KL-divergence distillation (Eq 4-6) between teacher LM response likelihoods $P_T(p \mid q, r)$ and retriever distribution $P_R(p \mid q)$.
3. **Weight Freezing**:
   - Generator/reader layers frozen; updates applied to $W_Q, W_K$ of layers $0 \dots b$ and adaptive routing modules.

---

## ⚡ Compute Savings vs Normal RAG

| System | Context Attention Compute | Retrieval Models Required |
| :--- | :--- | :--- |
| **Normal RAG** | All 32 layers process passage text tokens | 2 models (separate BERT/DPR + LLM) |
| **Baseline ImpRAG** | Only 8 middle layers ($L_M$: $b \dots t$) process passage KV — **~75% reduction** | 1 unified model |
| **Adaptive ImpRAG** | **75% reduction + parametric bypass (k=0) + adaptive k** — further savings | 1 unified model |

---

## 🚀 Quickstart & Verification

### 1. Run Verification Test Suites
Verify Baseline ImpRAG and all 4 dimensions of Adaptive ImpRAG:
```bash
python test_baseline_verification.py
python test_adaptive_verification.py
python test_capstone_adaptive_verification.py
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
