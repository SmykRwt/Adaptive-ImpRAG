# ImpRAG Base Implementation

This repo is a faithful base implementation of the paper **ImpRAG: Retrieval-Augmented Generation with Implicit Queries** using Hugging Face, PyTorch, and FAISS.

It includes:

- model slicing into `LB`, `LM`, and `LT`
- query embedding from the **last token attention query state**
- passage embedding from the **last token attention key state**
- FAISS inner-product retrieval
- independent passage encoding into per-layer passage states
- cross-attention injection through the middle layers
- query position shifting by `k * max_passage_length`
- warmup retrieval loss with pseudo positives and negatives
- self-distillation retrieval loss from generation log-likelihoods
- single-retrieval inference
- exact match and retrieval recall helpers

## Important scope

This is a **base implementation**, not a reproduction of the full Meta training stack.

What is implemented now:

- the paper’s end-to-end control flow
- the core losses and indexing pipeline
- the paper’s simplest passage encoding strategy: `independent`
- a single-retrieval decoding loop

What you will likely improve next:

- batched multi-example generation training
- segmented and full-attention passage encoding
- passage KV caching on disk
- decode-time KV caching for the query tokens
- distributed training and large-corpus retrieval services

## Install

```bash
pip install -r requirements.txt
```

If you want to use the exact model named in the paper setup notes:

```bash
huggingface-cli login
```

Then set:

```text
meta-llama/Llama-3.2-3B
```

If your GPU is weak or you do not have access to the Llama checkpoint yet, start with a smaller open model for debugging and then switch back.

## Repo layout

- [imprag/modeling_imprag.py](C:/Users/rawat/Documents/Codex/2026-04-25/files-mentioned-by-the-user-imprag/imprag/modeling_imprag.py)
- [imprag/retrieval.py](C:/Users/rawat/Documents/Codex/2026-04-25/files-mentioned-by-the-user-imprag/imprag/retrieval.py)
- [imprag/losses.py](C:/Users/rawat/Documents/Codex/2026-04-25/files-mentioned-by-the-user-imprag/imprag/losses.py)
- [imprag/data.py](C:/Users/rawat/Documents/Codex/2026-04-25/files-mentioned-by-the-user-imprag/imprag/data.py)
- [scripts/build_faiss_index.py](C:/Users/rawat/Documents/Codex/2026-04-25/files-mentioned-by-the-user-imprag/scripts/build_faiss_index.py)
- [scripts/train_imprag.py](C:/Users/rawat/Documents/Codex/2026-04-25/files-mentioned-by-the-user-imprag/scripts/train_imprag.py)
- [scripts/infer_imprag.py](C:/Users/rawat/Documents/Codex/2026-04-25/files-mentioned-by-the-user-imprag/scripts/infer_imprag.py)

## How the implementation maps to the paper

### Phase 1: Base model surgery

`ImpRAGModel` slices the decoder into:

- `LB = [0 ... b]`
- `LM = [b+1 ... t]` for continued query processing with passage cross-attention
- `LT = [t+1 ... N-1]`

The retrieval step stays outside the forward pass initially:

1. query goes through `LB`
2. query embedding is sent to FAISS
3. retrieved passages are encoded separately
4. `LM` reads them via cross-attention
5. `LT` finishes generation

### Phase 2: Retriever

For the query:

- run prompt through `LB`
- take the **last token**
- project with the final bottom layer attention `q_proj`
- average grouped-query heads down to key-value head count

For the passage:

- run passage through `LB`
- take the **last token**
- project with the final bottom layer attention `k_proj`

Similarity:

```text
score = dot(query_embedding, passage_embedding)
```

### Phase 3: FAISS

The retriever uses:

```python
faiss.IndexFlatIP(dim)
```

and normalizes embeddings before insertion/search.

### Phase 4: Passage encoding

This repo starts with the paper’s simplest strategy:

- `independent` passage encoding

Each retrieved passage is encoded separately, then its hidden states are flattened into per-layer passage memory used by the middle layers.

### Phase 5: Cross-attention

For each middle layer, the query hidden states attend to the retrieved passage states using the layer’s own projections:

- query uses `q_proj`
- passages use `k_proj` and `v_proj`
- output goes through `o_proj`

### Phase 6: Position shift

Query positions are shifted during reading by:

```text
k * max_passage_length
```

This is implemented in `decode_with_passages`.

### Phase 7-9: Training

The trainer supports:

- warmup retrieval loss with multi-label NCE
- self-distillation with KL divergence from generation log-likelihoods
- final joint loss:

```text
generation_loss + lambda * retrieval_loss
```

### Phase 10: Inference

The inference script follows the paper’s single-retrieval flow:

1. encode prompt with `LB`
2. retrieve top-k once
3. encode retrieved passages
4. decode with cross-attention
5. keep the same retrieved passages for later tokens

## Data format

Training JSONL rows look like this:

```json
{
  "query": "What is the capital of France?",
  "answer": "Paris",
  "positives": [0, 2],
  "hard_negatives": [1],
  "candidate_passages": [
    {"id": "p1", "text": "Paris is the capital of France."},
    {"id": "p2", "text": "Berlin is the capital of Germany."},
    {"id": "p3", "text": "The Eiffel Tower is in Paris."}
  ]
}
```

The `positives` and `hard_negatives` arrays index into `candidate_passages`.

## Quick start

### 1. Build an index

```bash
python scripts/build_faiss_index.py ^
  --model-name meta-llama/Llama-3.2-3B ^
  --passages examples/passages.jsonl ^
  --index-path outputs/demo.index ^
  --metadata-path outputs/demo_passages.json
```

### 2. Run a small training pass

```bash
python scripts/train_imprag.py ^
  --train-jsonl examples/train.jsonl ^
  --model-name meta-llama/Llama-3.2-3B ^
  --config-out outputs/imprag_config.json
```

### 3. Run inference

```bash
python scripts/infer_imprag.py ^
  --config outputs/imprag_config.json ^
  --index-path outputs/demo.index ^
  --metadata-path outputs/demo_passages.json ^
  --query "What is the capital of France?"
```

## Notes for real paper training

- Retrieval datasets from the paper: `NaturalQuestions`, `HotpotQA`
- Non-retrieval or instruction-tuning data: instruction following, summarization, QA, plus synthetic phrase denoising and sentence generation
- Paper defaults mentioned in the experiments:
  - `b = 7`
  - `t = 19` for Llama-3.2-3B
  - `t = 23` for Llama-3.1-8B
  - `top_k = 10`
  - warmup for the first `3` epochs out of `10`

## Caveats

- Hugging Face internals vary across versions, so if a specific Llama release changes decoder-layer signatures you may need a small compatibility patch in `modeling_imprag.py`.
- This code uses a simple decode loop rather than a production-grade KV-cached generator.
- The paper’s best passage encoding result is **full-attention concatenation**; this repo starts from **independent encoding** because it is much easier to debug.
