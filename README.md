# Adaptive ImpRAG

`Adaptive ImpRAG` is an upgrade-friendly research scaffold for comparing a faithful `ImpRAG` baseline against improved variants under the **same retrieval corpus, same model family, and same evaluation setup**.

This repo now supports:

- baseline `ImpRAG` with `independent` passage encoding
- `Adaptive ImpRAG` with `full_attention` passage encoding for passage reading
- JSONL passage corpora and FAISS retrieval indexes
- Wikipedia-style subset preparation for storage-limited experiments
- side-by-side evaluation and comparison scripts

## Fresh Machine Setup

If a collaborator clones this repo on another computer, they should be able to get running with:

```powershell
git clone <your-repo-url>
cd "Adaptive ImpRAG"
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1
```

Then they can either run the demo pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
```

or launch the browser UI:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1
```

Important:

- generated files in `outputs/` are intentionally ignored by git
- large corpora in `corpora/` are also ignored by git
- each collaborator should recreate indexes and trained checkpoints locally

## What database is used

In the original paper, the knowledge corpus is **Wikipedia**.

In this repo, the retrieval "database" is:

1. a passage corpus in JSONL format
2. a FAISS inner-product index built from that corpus

For real experiments, the intended corpus is a **Wikipedia-style passage collection**, ideally a KILT/DPR-style preprocessed version or a reproducible subset of it.

## Baseline vs Adaptive ImpRAG

Baseline config:
- [examples/tiny_baseline_config.json](C:/Users/rawat/Desktop/Adaptive ImpRAG/examples/tiny_baseline_config.json)

Plus config:
- [examples/tiny_plus_config.json](C:/Users/rawat/Desktop/Adaptive ImpRAG/examples/tiny_plus_config.json)

Open-model baseline config:
- [examples/open_baseline_config.json](C:/Users/rawat/Desktop/Adaptive ImpRAG/examples/open_baseline_config.json)

Open-model Adaptive config:
- [examples/open_plus_config.json](C:/Users/rawat/Desktop/Adaptive ImpRAG/examples/open_plus_config.json)

Main difference right now:

- baseline uses `passage_encoding_strategy = "independent"`
- `Adaptive ImpRAG` uses `passage_encoding_strategy = "full_attention"`

That matches the paper’s finding that full-attention passage encoding is stronger than independent encoding.

## Important files

Core package:

- [imprag/modeling_imprag.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/imprag/modeling_imprag.py)
- [imprag/retrieval.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/imprag/retrieval.py)
- [imprag/losses.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/imprag/losses.py)
- [imprag/metrics.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/imprag/metrics.py)
- [imprag/config.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/imprag/config.py)

Scripts:

- [scripts/prepare_wikipedia_subset.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/scripts/prepare_wikipedia_subset.py)
- [scripts/build_faiss_index.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/scripts/build_faiss_index.py)
- [scripts/train_imprag.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/scripts/train_imprag.py)
- [scripts/infer_imprag.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/scripts/infer_imprag.py)
- [scripts/evaluate_imprag.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/scripts/evaluate_imprag.py)
- [scripts/compare_runs.py](C:/Users/rawat/Desktop/Adaptive ImpRAG/scripts/compare_runs.py)

Toy/demo assets:

- [examples/passages.jsonl](C:/Users/rawat/Desktop/Adaptive ImpRAG/examples/passages.jsonl)
- [examples/train.jsonl](C:/Users/rawat/Desktop/Adaptive ImpRAG/examples/train.jsonl)

## Install

```powershell
python -m pip install -r requirements.txt
```

Set the repo root on `PYTHONPATH` before running the scripts:

```powershell
$env:PYTHONPATH='.'
```

These dependencies are now pinned to the versions tested in this repo so collaborators are less likely to hit environment drift.

## Recommended experiment plan

### Step 1. Create a Wikipedia-style subset

If the Hugging Face dataset is available, prepare a reproducible local subset:

```powershell
python scripts/prepare_wikipedia_subset.py `
  --dataset-name kilt_wikipedia `
  --split full `
  --sample-size 50000 `
  --seed 13 `
  --streaming `
  --append-title `
  --output-path corpora/wiki_subset_50k.jsonl
```

If `kilt_wikipedia` fails because the current `datasets` library no longer supports dataset scripts, the script now automatically falls back to:

- dataset: `wikimedia/wikipedia`
- config: `20231101.en`
- split: `train`

It also supports custom `--id-column`, `--text-column`, and `--title-column` if you point it at another compatible dataset source.

### Step 1b. Prepare a larger QA eval/train subset

For a real comparison, move beyond the 2-example toy eval file. For example, prepare a 100-example Natural Questions subset:

```powershell
python scripts/prepare_qa_dataset.py `
  --dataset-name nq_open `
  --split validation `
  --dataset-format nq_open `
  --sample-size 100 `
  --output-path corpora/nq_eval_100.jsonl
```

Or a HotpotQA subset:

```powershell
python scripts/prepare_qa_dataset.py `
  --dataset-name hotpot_qa `
  --dataset-config distractor `
  --split validation `
  --dataset-format hotpotqa `
  --sample-size 100 `
  --output-path corpora/hotpot_eval_100.jsonl
```

### Step 1c. Build retrieval-style training data from the harder corpus

Once you have a Wikipedia-style index and a QA subset, build candidate-passage examples for training:

```powershell
python scripts/build_retrieval_training_data.py `
  --config examples/open_baseline_config.json `
  --index-path outputs/wiki_subset_50k.index `
  --metadata-path outputs/wiki_subset_50k_meta.json `
  --qa-jsonl corpora/nq_eval_100.jsonl `
  --output-path corpora/nq_train_candidates_100.jsonl
```

This gives you a much more meaningful training/eval file than the toy handwritten examples.

### Step 2. Build the FAISS index

```powershell
python scripts/build_faiss_index.py `
  --config examples/tiny_baseline_config.json `
  --passages corpora/wiki_subset_50k.jsonl `
  --index-path outputs/wiki_subset_50k.index `
  --metadata-path outputs/wiki_subset_50k_meta.json
```

You can also build a small sampled index directly from a larger local JSONL:

```powershell
python scripts/build_faiss_index.py `
  --config examples/tiny_baseline_config.json `
  --passages corpora/wiki_subset_50k.jsonl `
  --sample-size 10000 `
  --seed 13 `
  --index-path outputs/wiki_subset_10k.index `
  --metadata-path outputs/wiki_subset_10k_meta.json
```

### Step 3. Train baseline ImpRAG

```powershell
python scripts/train_imprag.py `
  --config examples/tiny_baseline_config.json `
  --train-jsonl corpora/nq_train_candidates_100.jsonl `
  --epochs 1 `
  --warmup-epochs 1 `
  --config-out outputs/baseline_config.json
```

### Step 4. Train Adaptive ImpRAG

```powershell
python scripts/train_imprag.py `
  --config examples/tiny_plus_config.json `
  --train-jsonl corpora/nq_train_candidates_100.jsonl `
  --epochs 1 `
  --warmup-epochs 1 `
  --config-out outputs/plus_config.json
```

### Step 5. Evaluate baseline

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/evaluate_imprag.py `
  --config outputs/baseline_config.json `
  --index-path outputs/wiki_subset_50k.index `
  --metadata-path outputs/wiki_subset_50k_meta.json `
  --eval-jsonl corpora/nq_eval_100.jsonl `
  --output-path outputs/baseline_eval.json `
  --run-name baseline
```

### Step 6. Evaluate ImpRAG+

```powershell
python scripts/evaluate_imprag.py `
  --config outputs/plus_config.json `
  --index-path outputs/wiki_subset_50k.index `
  --metadata-path outputs/wiki_subset_50k_meta.json `
  --eval-jsonl corpora/nq_eval_100.jsonl `
  --output-path outputs/plus_eval.json `
  --run-name adaptive-imprag
```

### Step 7. Compare runs

```powershell
python scripts/compare_runs.py `
  --baseline outputs/baseline_eval.json `
  --candidate outputs/plus_eval.json
```

## Open model to use

For real local experiments without a gated checkpoint, use:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Why this model:

- it is open
- it uses the LLaMA architecture, which matches the current implementation
- it is much more meaningful than the random smoke-test model
- it is still small enough to be practical on limited hardware compared with 3B+ models

## How to run with the open model

### Build index

```powershell
python scripts/build_faiss_index.py `
  --config examples/open_baseline_config.json `
  --passages examples/passages.jsonl `
  --index-path outputs/open_demo.index `
  --metadata-path outputs/open_demo_meta.json
```

### Train baseline

```powershell
python scripts/train_imprag.py `
  --config examples/open_baseline_config.json `
  --train-jsonl examples/train.jsonl `
  --epochs 1 `
  --warmup-epochs 1 `
  --model-output-dir outputs/open_baseline `
  --config-out outputs/open_baseline_config_out.json
```

### Train Adaptive ImpRAG

```powershell
python scripts/train_imprag.py `
  --config examples/open_plus_config.json `
  --train-jsonl examples/train.jsonl `
  --epochs 1 `
  --warmup-epochs 1 `
  --model-output-dir outputs/open_plus `
  --config-out outputs/open_plus_config_out.json
```

### Inference with the open model

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/infer_imprag.py `
  --config outputs/open_plus_config_out.json `
  --index-path outputs/open_demo.index `
  --metadata-path outputs/open_demo_meta.json `
  --query "What is the capital of France?" `
  --top-k 2 `
  --max-new-tokens 16
```

The CLI now returns both:

- `answer_text`: raw generated answer
- `cleaned_answer_text`: short-answer cleaned output for factoid QA

And evaluation now reports both:

- `exact_match`: raw-answer exact match
- `cleaned_exact_match`: exact match after answer cleaning

## Browser UI

If you prefer not to type questions directly into the terminal, there is now a Streamlit app:

```powershell
streamlit run apps/streamlit_app.py
```

Then open the local URL Streamlit shows in the terminal, enter your question, and click `Compare Models`.

This is a good idea for your project because it makes testing much easier:

- you can type questions naturally
- you can compare baseline and Adaptive ImpRAG in one screen
- you can see retrieved passages beside each answer
- you can compare raw vs cleaned answers quickly
- it is much easier to demo than terminal-only input

## How to run the current tiny smoke test

### Build toy index

```powershell
python scripts/build_faiss_index.py `
  --config examples/tiny_baseline_config.json `
  --passages examples/passages.jsonl `
  --index-path outputs/demo.index `
  --metadata-path outputs/demo_passages.json
```

### Baseline training

```powershell
python scripts/train_imprag.py `
  --config examples/tiny_baseline_config.json `
  --train-jsonl examples/train.jsonl `
  --epochs 1 `
  --warmup-epochs 1 `
  --model-output-dir outputs/tiny_baseline `
  --config-out outputs/tiny_baseline_config_out.json
```

### Adaptive ImpRAG training

```powershell
python scripts/train_imprag.py `
  --config examples/tiny_plus_config.json `
  --train-jsonl examples/train.jsonl `
  --epochs 1 `
  --warmup-epochs 1 `
  --model-output-dir outputs/tiny_plus `
  --config-out outputs/tiny_plus_config_out.json
```

### Inference

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/infer_imprag.py `
  --config outputs/tiny_plus_config_out.json `
  --index-path outputs/demo.index `
  --metadata-path outputs/demo_passages.json `
  --query "What is the capital of France?" `
  --top-k 2 `
  --max-new-tokens 16
```

## Current limitations

- The training/eval examples are still toy examples; real comparison requires real retrieval datasets.
- The default tiny configs still use a random tiny Llama model only for smoke testing.
- For real results, use the new open configs or a gated checkpoint plus a Wikipedia-style corpus.
- The repo still uses simple greedy decoding and does not yet include production-grade KV caching.

## Best next move

For a meaningful `ImpRAG` vs `Adaptive ImpRAG` comparison:

1. prepare a fixed Wikipedia subset
2. create a small NQ/HotpotQA-style train/eval file against that same subset
3. train both baseline and plus on the same data
4. compare `Exact Match` and `Retrieval Recall@k`
5. pay special attention to `cleaned_exact_match` when raw generation is slightly noisy
