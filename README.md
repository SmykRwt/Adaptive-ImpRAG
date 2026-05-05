# Adaptive ImpRAG

Adaptive ImpRAG is a research scaffold for comparing:

- a baseline **ImpRAG** configuration
- an improved **Adaptive ImpRAG** configuration

under the same:

- retrieval corpus
- model family
- evaluation set

The current main improvement is:

- baseline: `passage_encoding_strategy = "independent"`
- adaptive: `passage_encoding_strategy = "full_attention"`

## What This Repo Contains

Core package:

- [`imprag/modeling_imprag.py`](imprag/modeling_imprag.py)
- [`imprag/retrieval.py`](imprag/retrieval.py)
- [`imprag/losses.py`](imprag/losses.py)
- [`imprag/metrics.py`](imprag/metrics.py)
- [`imprag/config.py`](imprag/config.py)

Scripts:

- [`scripts/prepare_wikipedia_subset.py`](scripts/prepare_wikipedia_subset.py)
- [`scripts/prepare_qa_dataset.py`](scripts/prepare_qa_dataset.py)
- [`scripts/build_faiss_index.py`](scripts/build_faiss_index.py)
- [`scripts/build_retrieval_training_data.py`](scripts/build_retrieval_training_data.py)
- [`scripts/train_imprag.py`](scripts/train_imprag.py)
- [`scripts/infer_imprag.py`](scripts/infer_imprag.py)
- [`scripts/evaluate_imprag.py`](scripts/evaluate_imprag.py)
- [`scripts/compare_runs.py`](scripts/compare_runs.py)

UI:

- [`apps/streamlit_app.py`](apps/streamlit_app.py)

Bootstrap helpers:

- [`scripts/bootstrap_env.ps1`](scripts/bootstrap_env.ps1)
- [`scripts/run_demo.ps1`](scripts/run_demo.ps1)
- [`scripts/run_streamlit.ps1`](scripts/run_streamlit.ps1)

Example configs:

- [`examples/open_baseline_config.json`](examples/open_baseline_config.json)
- [`examples/open_plus_config.json`](examples/open_plus_config.json)
- [`examples/tiny_baseline_config.json`](examples/tiny_baseline_config.json)
- [`examples/tiny_plus_config.json`](examples/tiny_plus_config.json)

## What Database Is Used

The original paper uses **Wikipedia** as the knowledge corpus.

In this repo, the retrieval “database” is:

1. a JSONL passage corpus
2. a FAISS inner-product index built from that corpus

For real experiments, use a Wikipedia-style corpus prepared with:

- [`scripts/prepare_wikipedia_subset.py`](scripts/prepare_wikipedia_subset.py)

## Fresh Machine Setup

After cloning the repo, a collaborator should be able to run:

```powershell
git clone <your-repo-url>
cd "Adaptive ImpRAG"
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1
```

Then either:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
```

Notes:

- generated artifacts in `outputs/` are ignored by git
- larger corpora in `corpora/` are ignored by git
- collaborators should recreate indexes and trained checkpoints locally

## Requirements

Install manually if you prefer:

```powershell
python -m pip install -r requirements.txt
```

Pinned dependencies are in [`requirements.txt`](requirements.txt) to reduce environment drift between machines.

## Open Model Used

For practical local experiments without a gated model, this repo uses:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Why:

- open access
- LLaMA-style architecture
- much more meaningful than a random smoke-test model
- still feasible on limited hardware compared with larger checkpoints

## Quick Start

### Option A: One-command demo

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
```

This will:

1. build a tiny FAISS demo index
2. train baseline on toy data
3. train Adaptive ImpRAG on toy data
4. run one inference example

### Option B: Browser UI

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
```

Then open the local URL shown by Streamlit and click `Compare Models`.

The UI compares:

- baseline ImpRAG
- Adaptive ImpRAG

side by side for the same question.

## CLI Inference

Once you have a trained config and an index, run:

```powershell
$env:PYTHONPATH='.'
$env:PYTHONIOENCODING='utf-8'
python scripts/infer_imprag.py `
  --config outputs/open_plus_config_out.json `
  --index-path outputs/open_demo.index `
  --metadata-path outputs/open_demo_meta.json `
  --query "What is the capital of France?" `
  --top-k 2 `
  --max-new-tokens 3
```

The result includes:

- `answer_text`: raw generated answer
- `cleaned_answer_text`: cleaned short answer for factoid QA
- `retrieved_passages`

## Real Experiment Workflow

### 1. Prepare a Wikipedia subset

Example:

```powershell
python scripts/prepare_wikipedia_subset.py `
  --dataset-name kilt_wikipedia `
  --split full `
  --sample-size 2000 `
  --seed 13 `
  --streaming `
  --append-title `
  --output-path corpora/wiki_subset_2k.jsonl
```

If `kilt_wikipedia` fails because the current `datasets` library no longer supports dataset-script repos, the script automatically falls back to:

- dataset: `wikimedia/wikipedia`
- config: `20231101.en`
- split: `train`

### 2. Build the FAISS index

```powershell
python scripts/build_faiss_index.py `
  --config examples/open_baseline_config.json `
  --passages corpora/wiki_subset_2k.jsonl `
  --index-path outputs/wiki_subset_2k.index `
  --metadata-path outputs/wiki_subset_2k_meta.json
```

### 3. Prepare a QA subset

Example with Natural Questions:

```powershell
python scripts/prepare_qa_dataset.py `
  --dataset-name nq_open `
  --split validation `
  --dataset-format nq_open `
  --sample-size 25 `
  --output-path corpora/nq_eval_25.jsonl
```

### 4. Build retrieval training data

```powershell
python scripts/build_retrieval_training_data.py `
  --config examples/open_baseline_config.json `
  --index-path outputs/wiki_subset_2k.index `
  --metadata-path outputs/wiki_subset_2k_meta.json `
  --qa-jsonl corpora/nq_eval_25.jsonl `
  --output-path corpora/nq_train_candidates_25.jsonl
```

### 5. Train baseline ImpRAG

```powershell
python scripts/train_imprag.py `
  --config examples/open_baseline_config.json `
  --train-jsonl corpora/nq_train_candidates_25.jsonl `
  --epochs 1 `
  --warmup-epochs 1 `
  --model-output-dir outputs/open_baseline_2k_25 `
  --config-out outputs/open_baseline_2k_25_config_out.json
```

### 6. Train Adaptive ImpRAG

```powershell
python scripts/train_imprag.py `
  --config examples/open_plus_config.json `
  --train-jsonl corpora/nq_train_candidates_25.jsonl `
  --epochs 1 `
  --warmup-epochs 1 `
  --model-output-dir outputs/open_plus_2k_25 `
  --config-out outputs/open_plus_2k_25_config_out.json
```

### 7. Evaluate baseline

```powershell
python scripts/evaluate_imprag.py `
  --config outputs/open_baseline_2k_25_config_out.json `
  --index-path outputs/wiki_subset_2k.index `
  --metadata-path outputs/wiki_subset_2k_meta.json `
  --eval-jsonl corpora/nq_eval_25.jsonl `
  --output-path outputs/open_baseline_2k_25_eval.json `
  --run-name original-imprag-2k-25
```

### 8. Evaluate Adaptive ImpRAG

```powershell
python scripts/evaluate_imprag.py `
  --config outputs/open_plus_2k_25_config_out.json `
  --index-path outputs/wiki_subset_2k.index `
  --metadata-path outputs/wiki_subset_2k_meta.json `
  --eval-jsonl corpora/nq_eval_25.jsonl `
  --output-path outputs/open_plus_2k_25_eval.json `
  --run-name adaptive-imprag-2k-25
```

### 9. Compare both runs

```powershell
python scripts/compare_runs.py `
  --baseline outputs/open_baseline_2k_25_eval.json `
  --candidate outputs/open_plus_2k_25_eval.json
```

## Metrics

Evaluation currently reports:

- `exact_match`
- `cleaned_exact_match`
- `retrieval_recall_at_k`

When generation is slightly noisy, `cleaned_exact_match` is often the more useful first comparison.

## Streamlit Comparison

The Streamlit app compares both models side by side and shows:

- cleaned answer
- raw answer
- prompt used
- retrieved passages
- top-passage score summary

Run:

```powershell
streamlit run apps/streamlit_app.py
```

## Current Limitations

- CPU training is slow, especially for the Adaptive configuration
- toy configs are only for smoke testing
- the repo still uses a simple greedy decoding loop
- current experiments are reduced-scale reproductions, not paper-scale training

## Recommended GitHub Practice

Commit:

- source code
- configs
- scripts
- Streamlit app
- README
- `.gitignore`

Do not rely on committing:

- `outputs/*`
- `corpora/*`
- `__pycache__/*`

Those should be regenerated locally after cloning.
