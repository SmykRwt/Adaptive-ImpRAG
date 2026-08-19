def add_chapter3_methodology(gen):
    # ==========================================
    # CHAPTER 3: METHODOLOGY ADOPTED
    # ==========================================
    gen.add_heading_1("CHAPTER 3: METHODOLOGY ADOPTED")
    
    gen.add_heading_2("3.1 Investigative Techniques")
    gen.add_p(
        "To rigorously design, optimize, and validate Adaptive ImpRAG, this project employs a tripartite investigative methodology encompassing Descriptive, Comparative, and Experimental research techniques:"
    )
    gen.add_p(
        "1. Descriptive Investigation: Formally models the mathematical dynamics of internal transformer layer activations across 32 decoder layers in Meta-Llama-3-8B. Analyzes activation representations across Grouped-Query Attention heads, hidden state norm distributions, and anisotropy within the implicit query representation space R^d at layer b = 7."
    )
    gen.add_p(
        "2. Comparative Investigation: Establishes a rigorous, apples-to-apples benchmark comparing: (a) Baseline ImpRAG (fixed b=7, t=23, static k=5, uniform GQA pooling); (b) Traditional Retrieve-then-Generate RAG (DPR + LLaMA-3); and (c) Adaptive ImpRAG across Exact Match (EM), F1 accuracy, end-to-end inference latency, GPU VRAM allocation, and compute efficiency on NaturalQuestions, TriviaQA, and HotpotQA."
    )
    gen.add_p(
        "3. Experimental Investigation: Constructs an empirical hypothesis-testing framework evaluating each adaptive intervention in isolation via ablation studies: (a) Dynamic Retrieval Gate classification threshold tuning; (b) Entropy vs top-margin dynamic k allocation thresholds; (c) Multi-hop iteration depth vs accuracy trade-offs; (d) LoRA rank r in {4, 8, 16, 32} parameter reduction curves; and (e) Document utility scoring redundancy penalties."
    )

    gen.add_heading_2("3.2 Proposed Solution")
    gen.add_p(
        "The proposed Adaptive ImpRAG architecture is realized through eight tightly integrated subsystems, designed to overcome the static limitations of implicit retrieval while preserving shared parameter space advantages:"
    )

    gen.add_heading_3("3.2.1 Knowledge Base Construction & Sentence-Boundary Semantic Chunking")
    gen.add_p(
        "Traditional RAG pipelines segment text by arbitrary character or word counts, slicing sentences mid-phrase and severing entity-attribute bindings. To eliminate this deficiency, Adaptive ImpRAG implements the AdaptiveSemanticChunker (imprag/chunking.py):\n"
        "• Sentence-Boundary Detection: Evaluates text using regex-based punctuation boundaries (?<=[.?!])\\s+(?=[A-Z0-9\"]) to ensure chunks terminate exclusively on complete grammatical propositions.\n"
        "• Sliding-Window Semantic Overlap: Groups sentences into windows of target size S_target = 150 words with a sliding overlap S_overlap = 40 words, guaranteeing that multi-sentence facts spanning adjacent paragraphs are never bifurcated.\n"
        "• Document Title Grounding: Prepends document/entity title headers [Title: Document Name] to every chunk, resolving pronoun ambiguities (e.g. 'He was born in...' -> '[Albert Einstein] He was born in...')."
    )

    gen.add_heading_3("3.2.2 Query Representation with Learned GQA Softmax Attention Pooling")
    gen.add_p(
        "In Grouped-Query Attention (GQA), Meta-Llama-3-8B groups g = 4 query heads for each KV head. Let q_{h, i} in R^{d_h} denote the i-th query head representation for KV head h in {1..8}. While baseline ImpRAG averages heads uniformly, Adaptive ImpRAG implements AdaptiveGQAPooling (imprag/adaptive.py), learning query-conditioned importance weights alpha_{h, i}(q):"
    )
    gen.add_callout(
        "alpha_{h, i}(q) = exp( W_h * q_{h, i} ) / sum_{j=1}^g exp( W_h * q_{h, j} )\n"
        "E_q = Concat_{h=1}^{h_k} ( sum_{i=1}^g alpha_{h, i}(q) * q_{h, i} )\n"
        "where W_h in R^{1 x d_h} is a learnable projection vector that prioritizes semantic heads and suppresses noise.",
        title="EQUATION 3.1: ADAPTIVE GQA ATTENTION HEAD POOLING"
    )

    gen.add_heading_3("3.2.3 Dynamic Retrieval Gate (Parametric Bypass Classifier)")
    gen.add_p(
        "To eliminate redundant vector searches on known facts, the DynamicRetrievalGate (imprag/adaptive.py) evaluates normalized query embedding E_norm = E_q / ||E_q||_2 at layer b = 7 through a two-layer feedforward network with sigmoid activation:"
    )
    gen.add_callout(
        "P_retrieve(q) = sigma( W_2 * ReLU( W_1 * E_norm + b_1 ) + b_2 )\n"
        "Decision: If P_retrieve(q) < 0.45 -> Bypasses FAISS search (k=0), executing direct generation and saving ~85% compute.",
        title="EQUATION 3.2: DYNAMIC RETRIEVAL DECISION GATE"
    )

    gen.add_heading_3("3.2.4 Adaptive k Budget Allocator (Entropy & Top-Margin Routing)")
    gen.add_p(
        "When retrieval is triggered, FAISS retrieves top candidates with inner product scores s_1 >= s_2 >= ... >= s_M. The AdaptiveKAllocator (imprag/adaptive.py) converts scores to probabilities p_i = exp(s_i / tau) / sum_j exp(s_j / tau) and computes the Shannon entropy H(p) and top margin Delta = s_1 - s_2:"
    )
    gen.add_bullet("If Delta >= 3.0 or H(p) <= 0.25: Dominant document detected -> Allocates k = 1 passage.", bold_prefix="• Clear Single-Fact Query:")
    gen.add_bullet("If 1.5 <= Delta < 3.0 or 0.25 < H(p) <= 0.55: Moderate confidence -> Allocates k = 2 passages.", bold_prefix="• Focused Multi-Sentence Query:")
    gen.add_bullet("If 0.6 <= Delta < 1.5 or 0.55 < H(p) <= 0.85: Moderate diffusion -> Allocates k = 5 passages.", bold_prefix="• Multi-Topic Factual Query:")
    gen.add_bullet("If Delta < 0.6 or H(p) > 0.85: Highly diffuse distribution -> Allocates k = 10 passages for full coverage.", bold_prefix="• Complex Ambiguous Query:")

    gen.add_heading_3("3.2.5 Adaptive Layer Boundary Router ([b(q), t(q)])")
    gen.add_p(
        "The AdaptiveLayerBoundaryRouter (imprag/adaptive.py) dynamically selects injection depth tiers using a linear classifier over query embedding E_q:\n"
        "• Tier 1: Shallow [b=4, t=14] — Keyword & simple entity lookups (low computational depth).\n"
        "• Tier 2: Standard [b=7, t=20] — Standard multi-sentence factual question answering.\n"
        "• Tier 3: Deep [b=7, t=26] — Complex multi-hop associative reasoning requiring deep cross-layer propagation."
    )

    gen.add_heading_3("3.2.6 Iterative Multi-Hop Retrieval Engine")
    gen.add_p(
        "For multi-hop queries, IterativeImpRAGRetriever (imprag/iterative.py) executes a multi-pass reasoning loop:\n"
        "• Hop 1: Computes initial query embedding E_q^{(1)}, retrieves candidates, and assesses context sufficiency.\n"
        "• Hop 2: If context sufficiency < 0.60 or multi-hop triggers are detected ('and', 'who also', 'after'), constructs an intermediate reasoning prompt, computes refined query embedding E_q^{(2)}, retrieves complementary evidence, and fuses passages into a unified concatenated KV cache."
    )

    gen.add_heading_3("3.2.7 Document Utility Scoring & Sufficiency Filtering")
    gen.add_p(
        "The DocumentUtilityScorer (imprag/utility.py) computes a combined utility score for each retrieved passage:\n"
        "U(p_i, q, P_selected) = [ 0.5 * DenseNorm(s_i) + 0.5 * LexicalRel(p_i, q) ] - lambda_redundancy * max_{p_j in P_selected} Jaccard(p_i, p_j)\n"
        "Passages falling below threshold tau_utility = 0.20 are discarded as distractors before KV injection."
    )

    gen.add_heading_3("3.2.8 Parameter-Efficient LoRA Fine-Tuning & Multi-Retriever Supervision")
    gen.add_p(
        "To enable single-GPU fine-tuning, native LoRALinear modules (imprag/peft_lora.py) wrap W_Q, W_K in layers 0..b with rank r = 16, alpha = 32: W_eff = W_frozen + (alpha / r) * (B * A). All remaining base LLM weights are frozen, reducing trainable parameters to 12,288 (>99.8% reduction). During training, MultiRetrieverEnsembleSupervisor (imprag/supervision.py) computes soft training targets by fusing dense similarity (45%), BM25 lexical overlap (30%), and answer substring grounding (25%)."
    )

    gen.add_heading_2("3.3 Work Breakdown Structure (WBS)")
    gen.add_p("The project is structured into four functional work packages across team members:")
    gen.add_bullet("WP1: Core Architecture & Dynamic Layer Router (Lead: Samyak Rawat & Saksham Gupta) — Implemented Dynamic Retrieval Gate, Adaptive K Allocator, Layer Boundary Router, and Adaptive GQA Pooling.", bold_prefix="• Work Package 1:")
    gen.add_bullet("WP2: Knowledge Base & Vector Indexing (Lead: Noor Tandon) — Developed AdaptiveSemanticChunker, built 4096-d FAISS index with Dual-Centering normalization.", bold_prefix="• Work Package 2:")
    gen.add_bullet("WP3: Iterative Multi-Hop & Utility Filtering (Lead: Arshia Anand) — Designed IterativeImpRAGRetriever, DocumentUtilityScorer, and context sufficiency verification.", bold_prefix="• Work Package 3:")
    gen.add_bullet("WP4: Evaluation, PEFT LoRA & UI Telemetry (Lead: Kunal Gupta) — Built LoRA adapter pipeline, automated evaluation benchmarks, and Gradio 2-mode web application.", bold_prefix="• Work Package 4:")

    gen.add_heading_2("3.4 Tools and Technology Stack Summary")
    gen.add_p("Table 3.1 details the operational software stack employed across all project modules:")
    tech_headers = ["Layer / Subsystem", "Software / Framework", "Version", "Specific Module / Script", "Functional Responsibility"]
    tech_rows = [
        ["Model Core", "PyTorch", "2.3.0", "imprag/model.py", "Transformer layer slicing, autograd flow, KV cache management."],
        ["4D Adaptivity", "Custom PyTorch", "1.0.0", "imprag/adaptive.py", "Dynamic gate, dynamic k allocator, layer boundary router, GQA pooling."],
        ["Multi-Hop Engine", "Custom PyTorch", "1.0.0", "imprag/iterative.py", "Iterative multi-pass retrieval loop and intermediate query refinement."],
        ["Utility Scorer", "NumPy / Regex", "1.26.4", "imprag/utility.py", "Semantic relevance, Jaccard novelty, and context sufficiency filtering."],
        ["Semantic Chunker", "Python 3.10", "3.10.12", "imprag/chunking.py", "Sentence-boundary regex splitting and title grounding."],
        ["PEFT Module", "Custom PyTorch", "1.0.0", "imprag/peft_lora.py", "Native LoRALinear adapters (r=16, alpha=32) for W_Q, W_K projections."],
        ["Vector Database", "FAISS-GPU", "1.8.0", "imprag/retriever.py", "Inner Product similarity search with Dual-Centering."],
        ["Supervision", "Custom PyTorch", "1.0.0", "imprag/supervision.py", "Multi-retriever ensemble target generation (dense+BM25+grounding)."],
        ["User Interface", "Gradio", "4.31.5", "app_web.py", "2-mode web application with public URL tunneling and telemetry badges."]
    ]
    gen.add_table(tech_headers, tech_rows, [1600, 1800, 1000, 2400, 3200])
    gen.add_page_break()

print("Part 3 defined successfully.")
