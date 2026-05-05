from pathlib import Path
import sys

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imprag import FaissRetriever, ImpRAGConfig, ImpRAGModel


st.set_page_config(page_title="Adaptive ImpRAG", page_icon="📚", layout="wide")
st.title("Adaptive ImpRAG")
st.caption("Compare baseline ImpRAG and Adaptive ImpRAG side by side.")


@st.cache_resource(show_spinner=True)
def load_system(config_path: str, index_path: str, metadata_path: str):
    config = ImpRAGConfig.from_json(config_path)
    model = ImpRAGModel(config)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    retriever = FaissRetriever.load(index_path, metadata_path)
    return model, retriever


def run_model(
    config_path: str,
    index_path: str,
    metadata_path: str,
    query: str,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
):
    model, retriever = load_system(config_path, index_path, metadata_path)
    return model.generate_with_retrieval(
        prompt=query,
        retriever=retriever,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        stop_on_repeat=True,
        clean_answer=True,
    )


default_baseline_config = str(ROOT / "outputs" / "open_baseline_config_out.json")
default_adaptive_config = str(ROOT / "outputs" / "open_plus_config_out.json")
default_index = str(ROOT / "outputs" / "open_demo.index")
default_meta = str(ROOT / "outputs" / "open_demo_meta.json")

with st.sidebar:
    st.header("Comparison Setup")
    baseline_config_path = st.text_input("Baseline Config JSON", value=default_baseline_config)
    adaptive_config_path = st.text_input("Adaptive Config JSON", value=default_adaptive_config)
    index_path = st.text_input("FAISS Index", value=default_index)
    metadata_path = st.text_input("Metadata JSON", value=default_meta)
    top_k = st.slider("Top-k passages", min_value=1, max_value=10, value=2)
    max_new_tokens = st.slider("Max new tokens", min_value=1, max_value=32, value=3)
    repetition_penalty = st.slider("Repetition penalty", min_value=1.0, max_value=2.0, value=1.15, step=0.05)
    no_repeat_ngram_size = st.slider("No repeat ngram size", min_value=0, max_value=4, value=2)

query = st.text_area("Question", value="What is the capital of France?", height=120)

if st.button("Compare Models", type="primary"):
    with st.spinner("Running baseline and Adaptive ImpRAG..."):
        baseline_result = run_model(
            baseline_config_path,
            index_path,
            metadata_path,
            query,
            top_k,
            max_new_tokens,
            repetition_penalty,
            no_repeat_ngram_size,
        )
        adaptive_result = run_model(
            adaptive_config_path,
            index_path,
            metadata_path,
            query,
            top_k,
            max_new_tokens,
            repetition_penalty,
            no_repeat_ngram_size,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline ImpRAG")
        st.success(baseline_result["cleaned_answer_text"] or "(empty)")
        st.caption("Cleaned answer")
        st.code(baseline_result["answer_text"] or "(empty)")
        st.caption("Raw answer")
        with st.expander("Prompt Used", expanded=False):
            st.code(baseline_result["prompt_text"])
        st.markdown("**Retrieved Passages**")
        for idx, hit in enumerate(baseline_result["retrieved_passages"], start=1):
            st.markdown(f"**{idx}. {hit['id']}**  |  score={hit['score']:.4f}")
            st.write(hit["text"])

    with col2:
        st.subheader("Adaptive ImpRAG")
        st.success(adaptive_result["cleaned_answer_text"] or "(empty)")
        st.caption("Cleaned answer")
        st.code(adaptive_result["answer_text"] or "(empty)")
        st.caption("Raw answer")
        with st.expander("Prompt Used", expanded=False):
            st.code(adaptive_result["prompt_text"])
        st.markdown("**Retrieved Passages**")
        for idx, hit in enumerate(adaptive_result["retrieved_passages"], start=1):
            st.markdown(f"**{idx}. {hit['id']}**  |  score={hit['score']:.4f}")
            st.write(hit["text"])

    st.subheader("Quick Comparison")
    st.table(
        [
            {
                "Model": "Baseline ImpRAG",
                "Cleaned Answer": baseline_result["cleaned_answer_text"],
                "Top Passage": baseline_result["retrieved_passages"][0]["id"] if baseline_result["retrieved_passages"] else "",
                "Top Score": f"{baseline_result['retrieved_passages'][0]['score']:.4f}" if baseline_result["retrieved_passages"] else "",
            },
            {
                "Model": "Adaptive ImpRAG",
                "Cleaned Answer": adaptive_result["cleaned_answer_text"],
                "Top Passage": adaptive_result["retrieved_passages"][0]["id"] if adaptive_result["retrieved_passages"] else "",
                "Top Score": f"{adaptive_result['retrieved_passages'][0]['score']:.4f}" if adaptive_result["retrieved_passages"] else "",
            },
        ]
    )
