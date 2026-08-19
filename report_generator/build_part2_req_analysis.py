def add_chapter2_requirement_analysis(gen):
    # ==========================================
    # CHAPTER 2: REQUIREMENT ANALYSIS
    # ==========================================
    gen.add_heading_1("CHAPTER 2: REQUIREMENT ANALYSIS")
    
    gen.add_heading_2("2.1 Literature Survey")
    
    gen.add_heading_3("2.1.1 Theory Associated With Problem Area")
    gen.add_p(
        "The theoretical foundation of Adaptive ImpRAG intersects Transformer Attention Mechanics, Dense Semantic Vector Retrieval, Information Entropy, and Low-Rank Matrix Decomposition:"
    )
    gen.add_p(
        "1. Transformer Attention and Grouped-Query Attention (GQA): In standard Multi-Head Attention (MHA), each query head interacts with a dedicated key-value head. To reduce KV cache memory during autoregressive decoding, Grouped-Query Attention (GQA, Ainslie et al., 2023; Touvron et al., 2023) groups g = h_q / h_k query heads per key-value head. In Meta-Llama-3-8B (h_q = 32, h_k = 8, g = 4, d_h = 128), query activations at layer b take the tensor shape [batch, h_k, g, d_h]. Squeezing this representation into a retrieval embedding requires pooling across the g dimension without discarding critical semantic signals."
    )
    gen.add_p(
        "2. Dense Vector Retrieval and Dual-Centering: Dense retrieval projects queries q and passages p into a continuous latent space R^d using neural encoders. Relevance is measured via Inner Product similarity s(q, p) = <E_q, E_p>. In transformer hidden states, representations often suffer from anisotropy (the 'cone effect'), where embeddings cluster in a narrow directional subspace. Dual-Centering resolves this by subtracting the empirical global query mean mu_q and passage mean mu_p: s_centered(q, p) = <E_q - mu_q, E_p - mu_p>."
    )
    gen.add_p(
        "3. Shannon Entropy of Score Distributions: To quantify retrieval uncertainty, candidate inner product scores are converted to a probability distribution via softmax: p_i = exp(s_i / tau) / sum_j exp(s_j / tau). The normalized Shannon entropy H(p) = -sum_i p_i ln(p_i) / ln(k_max) directly measures distributional diffusion. High entropy indicates ambiguity across multiple documents, necessitating a larger passage budget k."
    )

    gen.add_heading_3("2.1.2 Existing Systems and Solutions")
    gen.add_p(
        "A critical survey of prior architectures reveals the progression of retrieval-augmented generation:\n"
        "• Traditional RAG (Lewis et al., 2020): Pipeline architecture coupling BM25/DPR retriever with BART/T5 generator. Suffers from explicit prompt concatenation and disjoint parameter optimization.\n"
        "• Dense Passage Retriever (DPR, Karpukhin et al., 2020): Dual-encoder dense retriever trained with in-batch negatives. High retrieval recall but blind to downstream generator needs.\n"
        "• Baseline ImpRAG (Zhang et al., 2025): Unified implicit architecture slicing transformer layers. Fixes semantic gap but introduces single-pass rigidity, fixed k=5 compute waste, and full fine-tuning overhead.\n"
        "• Self-RAG (Asai et al., 2024): Generates inline reflection tokens to evaluate retrieval relevance. Operates strictly in single-pass settings and cannot handle implicit KV cache routing.\n"
        "• Corrective RAG (CRAG, Yan et al., 2024): Evaluates retrieved document quality via external evaluator. Adds significant multi-model latency overhead."
    )

    gen.add_heading_3("2.1.3 Research Findings for Existing Literature")
    gen.add_p("Table 2.1 presents a comprehensive literature survey compiled across project team members:")
    
    lit_headers = ["Roll No.", "Student Name", "Paper Title & Authors", "Tools / Technology", "Key Findings & Theoretical Takeaways", "Citation"]
    lit_rows = [
        [
            "102303142",
            "Noor Tandon",
            "ImpRAG: Retrieval-Augmented Generation with Implicit Queries (Zhang et al., 2025)",
            "PyTorch, LLaMA-3, FAISS, Transformer Slicing",
            "Demonstrated that lower transformer layers can act as query encoders and middle layers as implicit KV caches, closing the semantic gap.",
            "[3]"
        ],
        [
            "102303142",
            "Noor Tandon",
            "Retrieval-Augmented Generation for Knowledge-Intensive Tasks (Lewis et al., 2020)",
            "BART, DPR, FAISS, Dense Retrieval",
            "Pioneered RAG framework; proved that non-parametric memory reduces factual hallucination in open-domain QA.",
            "[2]"
        ],
        [
            "102303144",
            "Arshia Anand",
            "Sufficient Context: A New Lens on RAG Systems (Joren et al., 2025)",
            "Context Coverage Scorer, Mistral, LLaMA-2",
            "Proved that verifying context sufficiency before generation is essential to eliminate hallucinations in knowledge-intensive NLP.",
            "[1]"
        ],
        [
            "102303144",
            "Arshia Anand",
            "Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2024)",
            "Reflection Tokens, Critique Loss, LLaMA-2",
            "Showed that inline self-reflection improves factual grounding, though restricted to single-pass explicit generation.",
            "[6]"
        ],
        [
            "102303370",
            "Kunal Gupta",
            "LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2022)",
            "Low-Rank Decomposition, Transformer Projections",
            "Established that weight updates during fine-tuning have low intrinsic dimension, enabling >99% parameter reduction via rank decomposition.",
            "[4]"
        ],
        [
            "102303370",
            "Kunal Gupta",
            "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)",
            "4-bit NormalFloat, Double Quantization, Paged Optimizers",
            "Enabled fine-tuning of 33B/65B parameter LLMs on a single 48GB GPU without performance degradation.",
            "[5]"
        ],
        [
            "102303519",
            "Samyak Rawat",
            "GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)",
            "Grouped-Query Attention, KV Head Sharing",
            "Demonstrated that sharing KV heads across query head groups retains MHA quality while dramatically accelerating decoding speed.",
            "[7]"
        ],
        [
            "102303519",
            "Samyak Rawat",
            "Llama 3 Model Card and Technical Report (Meta AI, 2024)",
            "Meta-Llama-3-8B/70B, GQA, RoPE",
            "Detailed architectural specifications of LLaMA-3 (32 layers, 32 query heads, 8 KV heads, 4096 hidden dim) used as base LLM.",
            "[8]"
        ],
        [
            "102317256",
            "Saksham Gupta",
            "HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA (Yang et al., 2018)",
            "Multi-Hop Benchmark, Distractor Passages",
            "Standardized multi-hop reasoning evaluation; demonstrated that single-pass retrievers fail when reasoning chains span multiple documents.",
            "[9]"
        ],
        [
            "102317256",
            "Saksham Gupta",
            "Billion-scale similarity search with GPUs (Johnson et al., 2019)",
            "FAISS, GPU Similarity Search, IndexFlatIP",
            "Engineered ultra-fast exact and approximate nearest neighbor search on GPU hardware, enabling sub-millisecond vector retrieval.",
            "[10]"
        ]
    ]
    gen.add_table(lit_headers, lit_rows, [1200, 1500, 2400, 1600, 2600, 700])

    gen.add_heading_3("2.1.4 Problem Identified")
    gen.add_p(
        "Synthesizing the literature survey establishes the core architectural problem: Prior work either decoupled retrieval and generation (causing semantic mismatch), or unified them statically (causing single-pass rigidity, compute waste on parametric queries, and prohibitive fine-tuning overhead). No existing framework simultaneously unifies dynamic gating, adaptive passage budgeting, multi-hop iterative refinement, document utility filtering, and parameter-efficient adaptation within an implicit KV cache space."
    )

    gen.add_heading_3("2.1.5 Survey of Tools and Technologies Used")
    gen.add_p("Table 2.2 outlines the software frameworks and libraries selected for implementation:")
    tool_headers = ["Software / Library", "Version", "Functional Domain", "Selection Justification & System Role"]
    tool_rows = [
        ["PyTorch", "2.3.0+cu121", "Deep Learning Framework", "Core tensor computation, autograd gradient flow, and custom module development."],
        ["HuggingFace Transformers", "4.41.2", "LLM Model Management", "Loading Meta-Llama-3-8B weights, tokenizer management, and DynamicCache handling."],
        ["FAISS-GPU", "1.8.0", "Vector Similarity Search", "High-throughput inner product similarity indexing with Dual-Centering normalization."],
        ["Accelerate", "0.30.1", "Hardware Acceleration", "Optimized bfloat16 device mapping and low-CPU memory usage loading."],
        ["Gradio", "4.31.5", "Interactive Web UI", "Deployment of 2-mode comparison interface with public URL tunneling and live telemetry."],
        ["NumPy & SciPy", "1.26.4", "Numerical Mathematics", "Shannon entropy calculation, top-margin scoring, and Dual-Centering mean manipulation."]
    ]
    gen.add_table(tool_headers, tool_rows, [1800, 1200, 2200, 4800])

    gen.add_heading_2("2.2 Software Requirement Specification (SRS)")
    gen.add_heading_3("2.2.1 Introduction")
    gen.add_p(
        "2.2.1.1 Purpose: This SRS document specifies the complete functional and non-functional requirements for the Adaptive ImpRAG system, establishing rigorous engineering criteria for module integration, latency benchmarks, and evaluation.\n"
        "2.2.1.2 Intended Audience: Machine Learning Engineers, NLP Researchers, Capstone Evaluation Panels, and Enterprise AI Architects.\n"
        "2.2.1.3 Project Scope: Covers the complete pipeline from document ingestion, semantic chunking, and FAISS indexing to dynamic query routing, iterative multi-hop retrieval, LoRA adaptation, and interactive web visualization."
    )
    gen.add_heading_3("2.2.2 Overall Description")
    gen.add_p(
        "2.2.2.1 Product Perspective: Adaptive ImpRAG functions as an end-to-end, high-performance RAG system capable of running locally or on GPU server backends with public web access.\n"
        "2.2.2.2 Product Features: (1) Dynamic Retrieval Gate; (2) Entropy-driven Dynamic k Allocation; (3) Adaptive Layer Slicing; (4) Iterative Multi-Hop Reasoning Loop; (5) Document Utility Scoring; (6) Native LoRA Fine-Tuning; (7) Multi-Retriever Ensemble Supervision; and (8) Real-Time Telemetry Dashboard."
    )
    gen.add_heading_3("2.2.3 External Interface Requirements")
    gen.add_p(
        "2.2.3.1 User Interfaces: Interactive Gradio UI featuring natural language query input, 2-mode architecture selector (Adaptive ImpRAG vs Baseline ImpRAG), retrieval trigger dropdown, temperature slider, deterministic response display, retrieved passage inspector, and telemetry badges.\n"
        "2.2.3.2 Hardware Interfaces: NVIDIA GPU (CUDA Compute Capability >= 8.0, e.g., A100, RTX 3090/4090) with minimum 16GB VRAM.\n"
        "2.2.3.3 Software Interfaces: Linux OS (Ubuntu 22.04 LTS), Python 3.10+, PyTorch 2.3+, CUDA 12.1+."
    )
    gen.add_heading_3("2.2.4 Other Non-Functional Requirements")
    gen.add_p(
        "2.2.4.1 Performance Requirements: Parametric bypass latency < 50ms; single-hop retrieval latency < 280ms; multi-hop 2-pass latency < 520ms; peak GPU VRAM allocation < 18GB during generation.\n"
        "2.2.4.2 Safety Requirements: Safe fallback handling when FAISS returns no valid indices; strict error handling for out-of-bounds layer indices.\n"
        "2.2.4.3 Security Requirements: Local execution ensuring zero data transmission to external closed APIs; complete corpus isolation."
    )

    gen.add_heading_2("2.3 Cost Analysis")
    gen.add_p("Table 2.3 presents a computational and monetary cost comparison between baseline and adaptive implementations:")
    cost_headers = ["System Component / Process", "Baseline ImpRAG Requirement", "Adaptive ImpRAG Requirement", "Efficiency Gain & Cost Impact"]
    cost_rows = [
        ["Fine-Tuning VRAM Footprint", ">100 GB VRAM (Multi-GPU Cluster)", "16.2 GB VRAM (Single GPU)", "83.8% VRAM reduction via native LoRA (r=16)"],
        ["Trainable Parameters", "8,030,000,000 parameters (100%)", "12,288 parameters (0.0015%)", ">99.8% reduction in trainable weight storage"],
        ["Parametric Query Compute", "Full FAISS + KV Encode (k=5)", "Zero FAISS + Zero KV Encode (k=0)", "~85% compute and latency savings on factual queries"],
        ["Average KV Cache Memory", "Fixed 960 tokens / query", "192 - 384 tokens / query", "60% - 80% reduction in KV cache memory footprint"]
    ]
    gen.add_table(cost_headers, cost_rows, [2200, 2400, 2400, 3000])

    gen.add_heading_2("2.4 Risk Analysis")
    gen.add_p("Table 2.4 details potential operational risks and implemented mitigation strategies:")
    risk_headers = ["Identified Risk", "Severity", "Likelihood", "Impact on System", "Mitigation Strategy Implemented in Adaptive ImpRAG"]
    risk_rows = [
        ["GPU Out-Of-Memory (OOM)", "High", "Low", "Inference crash during multi-hop KV concatenation", "Dynamic layer boundaries [b, t] and strict passage length capping at 192 tokens."],
        ["Multi-Hop Semantic Drift", "Medium", "Medium", "Second hop retrieves irrelevant tangential context", "Document Utility Scorer applies Jaccard novelty penalty and context sufficiency checks."],
        ["Corpus Representation Anisotropy", "Medium", "High", "Query and passage embeddings cluster in narrow cone", "Dual-Centering removes global query/passage means before FAISS inner product search."],
        ["Non-Deterministic Generation", "High", "Low", "Identical queries yield varying factual answers", "Default temperature=0.0 greedy decoding with early EOS token stopping."],
        ["Noisy Pseudo-Label Supervision", "Medium", "Medium", "Retriever learns inductive errors from single teacher", "Multi-Retriever Ensemble Supervisor fuses dense, lexical, and string-matching targets."]
    ]
    gen.add_table(risk_headers, risk_rows, [1800, 1000, 1000, 2600, 3600])
    gen.add_page_break()

print("Part 2 defined successfully.")
