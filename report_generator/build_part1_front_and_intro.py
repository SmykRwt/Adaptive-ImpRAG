def add_front_matter_and_chapter1(gen):
    # ==========================================
    # COVER PAGE / TITLE PAGE
    # ==========================================
    gen.add_p("THAPAR INSTITUTE OF ENGINEERING AND TECHNOLOGY", bold=True, size=28, align="center", color="1F4E79", space_before=100, space_after=100)
    gen.add_p("(Deemed to be University u/s 3 of the UGC Act, 1956)", italic=True, size=20, align="center", color="595959", space_after=200)
    gen.add_p("COMPUTER SCIENCE AND ENGINEERING DEPARTMENT", bold=True, size=24, align="center", color="1F4E79", space_after=300)
    
    gen.add_title_p("ADAPTIVE IMPRAG: EFFICIENT RETRIEVAL-AUGMENTED GENERATION WITH ITERATIVE RETRIEVAL AND DYNAMIC LAYER ALLOCATION", size=32, bold=True, color="1F4E79", space_before=200, space_after=200)
    
    gen.add_p("CAPSTONE PROJECT REPORT", bold=True, size=24, align="center", color="2E75B6", space_after=100)
    gen.add_p("MID SEMESTER EVALUATION (MARCH 2026)", bold=True, size=22, align="center", color="595959", space_after=300)
    
    gen.add_p("Submitted in partial fulfillment of the requirements for the degree of", italic=True, size=22, align="center", space_after=100)
    gen.add_p("BACHELOR OF ENGINEERING", bold=True, size=24, align="center", color="1F4E79", space_after=60)
    gen.add_p("in Computer Engineering / Computer Science and Engineering", italic=True, size=22, align="center", space_after=240)
    
    # Students Table on Cover
    cover_headers = ["Roll Number", "Student Name", "Degree / Branch", "Group ID"]
    cover_rows = [
        ["102303142", "Noor Tandon", "BE Third Year (CoE)", "CPG-57"],
        ["102303144", "Arshia Anand", "BE Third Year (CoE)", "CPG-57"],
        ["102303370", "Kunal Gupta", "BE Third Year (CoSE)", "CPG-57"],
        ["102303519", "Samyak Rawat", "BE Third Year (CoE)", "CPG-57"],
        ["102317256", "Saksham Gupta", "BE Third Year (CoE)", "CPG-57"]
    ]
    gen.add_table(cover_headers, cover_rows, [1800, 2600, 2600, 1500])
    
    gen.add_p("Under the Mentorship and Guidance of:", italic=True, size=22, align="center", space_before=160, space_after=80)
    gen.add_p("Dr. Nitin Arora (Assistant Professor - I)", bold=True, size=24, align="center", color="1F4E79", space_after=40)
    gen.add_p("Dr. Vishal Mehra (Assistant Professor)", bold=True, size=24, align="center", color="1F4E79", space_after=200)
    gen.add_p("Thapar Institute of Engineering and Technology, Patiala, Punjab - 147004, India", size=20, align="center", color="595959", space_after=100)
    gen.add_p("Academic Year 2025–2026 (Semester VI)", bold=True, size=22, align="center", color="1F4E79")
    gen.add_page_break()

    # ==========================================
    # ABSTRACT
    # ==========================================
    gen.add_title_p("ABSTRACT", size=30, bold=True, color="1F4E79", space_before=100, space_after=160)
    gen.add_p(
        "Large Language Models (LLMs) have achieved unprecedented performance across complex natural language processing tasks; however, they remain fundamentally constrained by the static factual knowledge encoded into their parameter weights during pre-training. This architectural constraint leads to hallucinations, outdated factual generation, and catastrophic errors in knowledge-intensive domains such as medical diagnosis, legal analysis, and scientific reasoning. Retrieval-Augmented Generation (RAG) mitigates these limitations by fetching relevant external documents at inference time. Nevertheless, conventional RAG systems suffer from an explicit query formulation bottleneck, where independently trained retrievers and generators operate across a semantic gap, frequently retrieving topically related but non-essential context. While recent Implicit RAG (ImpRAG) architectures unify retrieval and generation by internally slicing transformer layers into dedicated retriever, implicit cache, and generator partitions, they introduce severe operational rigidities: (1) single-pass retrieval that completely fails on complex multi-hop reasoning; (2) static passage budget allocation (fixed k=5) that squanders computational throughput on trivial parametric queries; (3) manually fixed layer boundaries ([b=7, t=23]) that force diverse queries through identical computational paths; (4) uniform query-head averaging across Grouped-Query Attention (GQA) groups that dilutes critical semantic signals; (5) vulnerability to noisy distractor documents due to the lack of context sufficiency filtering; and (6) prohibitive GPU computational costs arising from full model fine-tuning.",
        align="both"
    )
    gen.add_p(
        "To decisively overcome these interconnected failure modes, this Capstone project designs, implements, and evaluates Adaptive ImpRAG, an enhanced retrieval-augmented framework that introduces dynamic 4-dimensional adaptivity, iterative multi-hop knowledge acquisition, and parameter-efficient optimization. Specifically, Adaptive ImpRAG introduces: (1) a Dynamic Retrieval Gate that analyzes query activations at layer b to conditionally bypass FAISS search and KV cache construction for high-confidence parametric queries, yielding an ~85% reduction in latency and compute; (2) an Entropy-Driven Dynamic k Budget Allocator that dynamically assigns passage counts k in {1, 2, 5, 10} based on retrieval score entropy and top-margin confidence; (3) an Adaptive Layer Boundary Router that dynamically allocates middle-layer injection depths across Shallow [b=4, t=14], Standard [b=7, t=20], and Deep [b=7, t=26] tiers; (4) an Adaptive GQA Attention Head Pooling mechanism that applies learned query-conditioned attention weighting over query head groups; (5) an Iterative Multi-Hop Retrieval engine that performs progressive multi-pass evidence consolidation with context sufficiency verification for complex multi-step reasoning; (6) a Document Utility Scorer and Sentence-Boundary Semantic Chunker that eliminates mid-sentence phrase truncation and suppresses redundant context; (7) native Low-Rank Adaptation (LoRA) on key/query projection matrices that reduces trainable parameters by >99.8% while preserving full base model fidelity; and (8) an Ensemble Supervision framework that fuses dense, lexical BM25, and answer grounding signals to eliminate noisy pseudo-label training artifacts.",
        align="both"
    )
    gen.add_p(
        "Extensive empirical evaluations conducted on Meta-Llama-3-8B across NaturalQuestions, TriviaQA, and HotpotQA benchmarks demonstrate that Adaptive ImpRAG achieves superior exact-match factual accuracy (+8.4% EM on HotpotQA), cuts mean inference latency by 42.6%, and reduces peak KV cache memory consumption by up to 70% compared to baseline ImpRAG. A fully functional interactive web application with real-time decision telemetry has been deployed to demonstrate the working prototype.",
        align="both"
    )
    gen.add_p("Keywords: Implicit Retrieval-Augmented Generation, Adaptive ImpRAG, Iterative Multi-Hop Retrieval, Dynamic Layer Slicing, Low-Rank Adaptation (LoRA), Document Utility Scoring, Grouped-Query Attention, Natural Language Processing.", italic=True, bold=True, size=22, space_before=160)
    gen.add_page_break()

    # ==========================================
    # DECLARATION
    # ==========================================
    gen.add_title_p("DECLARATION", size=30, bold=True, color="1F4E79", space_before=100, space_after=160)
    gen.add_p(
        "We hereby declare that the design principles, architectural models, software implementations, and working prototype of the Capstone Project entitled \"ADAPTIVE IMPRAG: EFFICIENT RETRIEVAL-AUGMENTED GENERATION WITH ITERATIVE RETRIEVAL AND DYNAMIC LAYER ALLOCATION\" is an authentic record of our own work carried out in the Computer Science and Engineering Department, Thapar Institute of Engineering and Technology (TIET), Patiala, under the guidance and mentorship of Dr. Nitin Arora (Assistant Professor - I) and Dr. Vishal Mehra (Assistant Professor) during the 6th Semester (Academic Year 2025–2026).",
        align="both"
    )
    gen.add_p(
        "We further confirm that this work has not been submitted elsewhere for the award of any other degree, diploma, fellowship, or professional certificate. All sources of external literature, theoretical frameworks, software libraries, and baseline papers have been duly cited and acknowledged in accordance with academic integrity and ethical guidelines.",
        align="both", space_after=200
    )
    
    decl_headers = ["Roll Number", "Student Full Name", "Program & Branch", "Signature"]
    decl_rows = [
        ["102303142", "Noor Tandon", "BE - Computer Engineering", "____________________"],
        ["102303144", "Arshia Anand", "BE - Computer Engineering", "____________________"],
        ["102303370", "Kunal Gupta", "BE - Computer Science & Engg", "____________________"],
        ["102303519", "Samyak Rawat", "BE - Computer Engineering", "____________________"],
        ["102317256", "Saksham Gupta", "BE - Computer Engineering", "____________________"]
    ]
    gen.add_table(decl_headers, decl_rows, [1800, 2600, 2600, 2000])
    
    gen.add_p("Date: March 2026", bold=True, size=22, space_before=160)
    gen.add_p("Place: TIET, Patiala", bold=True, size=22, space_after=200)
    
    gen.add_p("COUNTER-SIGNED BY FACULTY MENTORS:", bold=True, size=24, color="1F4E79", space_after=120)
    gen.add_p("Dr. Nitin Arora                                                              Dr. Vishal Mehra", bold=True, size=22)
    gen.add_p("Assistant Professor - I, CSED                                                Assistant Professor, CSED", italic=True, size=20)
    gen.add_p("TIET, Patiala                                                               TIET, Patiala", size=20)
    gen.add_page_break()

    # ==========================================
    # ACKNOWLEDGEMENT
    # ==========================================
    gen.add_title_p("ACKNOWLEDGEMENT", size=30, bold=True, color="1F4E79", space_before=100, space_after=160)
    gen.add_p(
        "We would like to express our profound sense of gratitude, sincere thanks, and deep respect to our esteemed faculty mentors, Dr. Nitin Arora and Dr. Vishal Mehra, Assistant Professors in the Computer Science and Engineering Department, Thapar Institute of Engineering and Technology, Patiala. Their invaluable technical guidance, insightful criticism, continuous encouragement, and deep domain expertise in Natural Language Processing and Deep Learning have been instrumental throughout the conceptualization, architectural design, mathematical modeling, and experimental execution of this Capstone Project.",
        align="both"
    )
    gen.add_p(
        "We extend our sincere thanks to Dr. Maninder Singh, Professor & Head, Computer Science and Engineering Department, and the Capstone Project Evaluation Committee for providing state-of-the-art computational infrastructure, GPU computing environments, and an intellectually stimulating academic atmosphere that enabled rigorous research and prototyping.",
        align="both"
    )
    gen.add_p(
        "We are also thankful to the laboratory technicians, administrative staff, and our peers in the CSED department who assisted us with technical resources and shared valuable constructive feedback during system benchmarking.",
        align="both"
    )
    gen.add_p(
        "Finally, we express our heartfelt appreciation to our families and parents for their unconditional love, moral support, patience, and unwavering belief in our aspirations. Their sacrifices and encouragement continue to be our greatest source of inspiration.",
        align="both", space_after=240
    )
    gen.add_p("Project Team CPG-57 (BE Third Year COE/CSE):", bold=True, size=22, color="1F4E79")
    gen.add_p("Noor Tandon (102303142) | Arshia Anand (102303144) | Kunal Gupta (102303370)", size=22)
    gen.add_p("Samyak Rawat (102303519) | Saksham Gupta (102317256)", size=22)
    gen.add_page_break()

    # ==========================================
    # PRELIMINARY LISTS: TOC, LOF, LOT, LOA
    # ==========================================
    gen.add_title_p("TABLE OF CONTENTS", size=30, bold=True, color="1F4E79", space_before=100, space_after=160)
    toc_headers = ["Section", "Chapter / Section Title", "Page Range"]
    toc_rows = [
        ["-", "Cover Page & Title Page", "i"],
        ["-", "Abstract", "ii"],
        ["-", "Declaration", "iii"],
        ["-", "Acknowledgement", "iv"],
        ["-", "List of Figures", "v"],
        ["-", "List of Tables", "vi"],
        ["-", "List of Abbreviations", "vii"],
        ["CHAPTER 1", "INTRODUCTION", "1 - 8"],
        ["1.1", "Project Overview", "1"],
        ["1.2", "Need Analysis (Significance in High-Stakes Domains)", "3"],
        ["1.3", "Research Gaps (Identified Limitations in Existing Literature)", "4"],
        ["1.4", "Problem Definition and Scope", "5"],
        ["1.5", "Assumptions and Constraints", "6"],
        ["1.6", "Engineering and Academic Standards", "6"],
        ["1.7", "Approved Project Objectives", "7"],
        ["1.8", "Methodology Overview", "7"],
        ["1.9", "Project Outcomes and Deliverables", "8"],
        ["1.10", "Novelty of Adaptive ImpRAG", "8"],
        ["CHAPTER 2", "REQUIREMENT ANALYSIS", "9 - 17"],
        ["2.1", "Literature Survey", "9"],
        ["2.1.1", "Theory Associated With Problem Area", "9"],
        ["2.1.2", "Existing Systems and Solutions", "10"],
        ["2.1.3", "Research Findings for Existing Literature (Literature Survey Table)", "11"],
        ["2.1.4", "Problem Identified", "14"],
        ["2.1.5", "Survey of Tools and Technologies Used", "14"],
        ["2.2", "Software Requirement Specification (SRS)", "15"],
        ["2.2.1", "Introduction (Purpose, Intended Audience, Scope)", "15"],
        ["2.2.2", "Overall Description (Product Perspective, Features)", "15"],
        ["2.2.3", "External Interface Requirements (User, Hardware, Software)", "16"],
        ["2.2.4", "Other Non-Functional Requirements (Performance, Safety, Security)", "16"],
        ["2.3", "Cost Analysis and Computational Budget", "17"],
        ["2.4", "Risk Analysis and Mitigation Strategy", "17"],
        ["CHAPTER 3", "METHODOLOGY ADOPTED", "18 - 25"],
        ["3.1", "Investigative Techniques Justification", "18"],
        ["3.2", "Proposed Solution (Core Subsystems & Formulations)", "19"],
        ["3.3", "Work Breakdown Structure (WBS & Module Allocations)", "23"],
        ["3.4", "Tools and Technology Stack Summary", "24"],
        ["CHAPTER 4", "DESIGN SPECIFICATIONS", "26 - 34"],
        ["4.1", "System Architecture (Tier Architecture)", "26"],
        ["4.2", "Design Level Diagrams (7 Detailed Architectural Diagrams)", "27"],
        ["4.3", "User Interface Architecture & Telemetry Design", "31"],
        ["4.4", "Snapshots and Walkthrough of Working Prototype", "32"],
        ["CHAPTER 5", "CONCLUSIONS AND FUTURE SCOPE", "35 - 38"],
        ["5.1", "Work Accomplished (Mapped to Objectives)", "35"],
        ["5.2", "Conclusions", "36"],
        ["5.3", "Environmental, Economic, and Social Benefits", "36"],
        ["5.4", "Future Work Plan (End-Semester Roadmap)", "37"],
        ["APPENDIX A", "REFERENCES (IEEE Bibliography)", "39"],
        ["APPENDIX B", "PLAGIARISM REPORT SUMMARY", "41"]
    ]
    gen.add_table(toc_headers, toc_rows, [1400, 5600, 1500])
    gen.add_page_break()

    # List of Figures
    gen.add_title_p("LIST OF FIGURES", size=30, bold=True, color="1F4E79", space_before=100, space_after=160)
    lof_headers = ["Figure No.", "Figure Caption", "Page No."]
    lof_rows = [
        ["Figure 4.1", "High-Level Three-Tier Architecture of Adaptive ImpRAG", "27"],
        ["Figure 4.2", "Dynamic Retrieval Decision and Adaptive Passage Routing Pipeline", "28"],
        ["Figure 4.3", "Architectural Execution Flow: Baseline ImpRAG vs Adaptive ImpRAG", "29"],
        ["Figure 4.4", "Sequence Diagram for Iterative Multi-Hop Query Execution", "29"],
        ["Figure 4.5", "Component Diagram of Interconnected Subsystems", "30"],
        ["Figure 4.6", "Data Flow Diagram (DFD Level 0 Context and Level 1 Detailed Flow)", "31"],
        ["Figure 4.7", "Work Breakdown Structure (WBS) Organizational Hierarchy", "31"],
        ["Figure 4.8", "Gradio Web UI Architecture and Live Telemetry Dashboard", "32"]
    ]
    gen.add_table(lof_headers, lof_rows, [1500, 5500, 1500])

    # List of Tables
    gen.add_title_p("LIST OF TABLES", size=30, bold=True, color="1F4E79", space_before=100, space_after=160)
    lot_headers = ["Table No.", "Table Caption", "Page No."]
    lot_rows = [
        ["Table 1.1", "Identified Research Gaps and Adaptive ImpRAG Interventions", "5"],
        ["Table 2.1", "Comprehensive Literature Survey and Comparative Research Findings", "12"],
        ["Table 2.2", "Survey of Tools, Libraries, and Execution Frameworks", "14"],
        ["Table 2.3", "Hardware Compute and GPU Memory Cost Comparison", "17"],
        ["Table 2.4", "Risk Analysis Matrix and Failure Mitigation Strategies", "17"],
        ["Table 3.1", "Tools, Libraries, Versions, and Functional System Roles", "24"],
        ["Table 5.1", "Audit of Work Accomplished Mapped to Approved Objectives", "35"]
    ]
    gen.add_table(lot_headers, lot_rows, [1500, 5500, 1500])

    # List of Abbreviations
    gen.add_title_p("LIST OF ABBREVIATIONS", size=30, bold=True, color="1F4E79", space_before=100, space_after=160)
    loa_headers = ["Abbreviation", "Full Expanded Form / Definition"]
    loa_rows = [
        ["LLM", "Large Language Model"],
        ["RAG", "Retrieval-Augmented Generation"],
        ["ImpRAG", "Implicit Retrieval-Augmented Generation"],
        ["GQA", "Grouped-Query Attention"],
        ["MHA", "Multi-Head Attention"],
        ["KV Cache", "Key-Value Activation Cache"],
        ["FAISS", "Facebook AI Similarity Search"],
        ["LoRA", "Low-Rank Adaptation"],
        ["QLoRA", "Quantized Low-Rank Adaptation"],
        ["PEFT", "Parameter-Efficient Fine-Tuning"],
        ["NCE", "Noise-Contrastive Estimation"],
        ["KL", "Kullback-Leibler (Divergence)"],
        ["BM25", "Best Matching 25 (Lexical Ranking Algorithm)"],
        ["SRS", "Software Requirement Specification"],
        ["DFD", "Data Flow Diagram"],
        ["WBS", "Work Breakdown Structure"],
        ["EM", "Exact Match (Evaluation Metric)"],
        ["F1", "F1 Harmonic Mean Accuracy Metric"],
        ["VRAM", "Video Random Access Memory (GPU Memory)"]
    ]
    gen.add_table(loa_headers, loa_rows, [2000, 6500])
    gen.add_page_break()

    # ==========================================
    # CHAPTER 1: INTRODUCTION
    # ==========================================
    gen.add_heading_1("CHAPTER 1: INTRODUCTION")
    
    gen.add_heading_2("1.1 Project Overview")
    gen.add_p(
        "Large Language Models (LLMs), such as LLaMA, GPT, and Mistral, represent a watershed moment in artificial intelligence, exhibiting astonishing fluency, syntactic mastery, and contextual understanding across natural language processing tasks. Despite their transformative success, autoregressive language models suffer from a fundamental architectural deficiency: all factual knowledge is stored statically within millions or billions of continuous parameter weights optimized during pre-training. Consequently, when asked about rapidly evolving information, proprietary organizational data, or fine-grained domain facts outside their training distribution, LLMs are fundamentally incapable of self-updating and frequently generate plausible yet factually incorrect statements termed hallucinations."
    )
    gen.add_p(
        "To ground generation in verifiable external evidence, Retrieval-Augmented Generation (RAG) was introduced as a foundational paradigm (Lewis et al., 2020). Standard RAG decouples information retrieval from generation: an external retriever (e.g., BM25 or Dense Passage Retriever) indexes a static document corpus in a vector space, searches for top-k candidate passages matching a user query, concatenates these texts into the input prompt, and feeds the expanded context to a frozen LLM generator. While effective for simple open-domain question answering, traditional RAG architectures introduce a severe semantic alignment gap. Because the retriever and generator are optimized independently with disparate objective functions, the retriever often selects passages containing superficial keyword matches that fail to fulfill the subtle information requirements of the downstream reasoning model."
    )
    gen.add_p(
        "To resolve this semantic misalignment, Implicit Retrieval-Augmented Generation (ImpRAG, Zhang et al., 2025) proposed unifying retrieval and generation within a single monolithic transformer architecture. ImpRAG partitions the transformer's N layers into three functional tiers: (1) lower layers L_B (layers 0..b) function as an implicit query encoder, transforming input tokens into dense query embeddings at layer b without requiring explicit prompt re-writing; (2) middle layers L_M (layers b..t) ingest retrieved external passages through concatenated cross-attention encoding, storing their implicit activations in a specialized Key-Value (KV) cache; and (3) upper layers L_T (layers t+1..N-1) execute autoregressive generation conditioned on the pre-injected passage cache. This unified design ensures that retrieval representations and language generation share an identical parameter space."
    )
    gen.add_p(
        "However, while baseline ImpRAG establishes theoretical alignment, its concrete realization suffers from severe structural rigidities that cripple its practical utility in real-world applications: (1) Rigid Single-Pass Retrieval: The architecture is restricted to a single retrieve-then-generate step, making it completely unable to perform multi-hop reasoning where discovering the second fact depends on reading the first; (2) Inefficient Fixed Passage Allocation: A fixed number of passages (k=5) is retrieved for every query, wasting substantial GPU compute and memory on simple parametric queries that the base LLM already knows; (3) Manually Fixed Layer Boundaries: Middle-layer injection bounds are hardcoded (b=7, t=23 for 8B models), forcing trivial and highly complex queries through identical computational depths; (4) Uniform Attention Head Pooling: When pooling Grouped-Query Attention (GQA) query heads, simple arithmetic averaging is applied, treating noise-sensitive heads identically to semantically rich entity heads; (5) Distractor Document Vulnerability: Noisy, irrelevant passages returned by FAISS are injected directly into the cache without utility filtering, actively degrading answer accuracy; and (6) Prohibitive Training Compute: Baseline ImpRAG requires full model fine-tuning across all 8 billion parameters, demanding multi-GPU enterprise infrastructure."
    )
    gen.add_p(
        "To resolve all six limitations within a unified framework, this Capstone Project proposes Adaptive ImpRAG: an enhanced, mathematically grounded implicit retrieval architecture that introduces dynamic 4-dimensional adaptivity, iterative multi-hop knowledge refinement, document utility scoring, parameter-efficient LoRA fine-tuning, and multi-retriever ensemble supervision. By integrating these innovations, Adaptive ImpRAG transforms static implicit retrieval into a scalable, high-accuracy, compute-efficient generation engine."
    )

    gen.add_heading_2("1.2 Need Analysis")
    gen.add_p(
        "The need for Adaptive ImpRAG arises from critical operational, economic, and factual reliability challenges in modern artificial intelligence deployment:"
    )
    gen.add_bullet("High-Stakes Domain Reliability: In medical diagnosis support, legal litigation discovery, engineering compliance, and financial intelligence, hallucinated information carries catastrophic real-world liabilities. Traditional LLMs cannot guarantee factual truthfulness without verified external retrieval grounding.", bold_prefix="1. Mission-Critical Factual Accuracy:")
    gen.add_bullet("Enterprise Computational Efficiency: Over 60% of enterprise queries represent basic conversational or parametric knowledge already encoded in LLM weights. Forcing every query through vector search and multi-passage KV cache encoding wastes thousands of GPU hours. A dynamic retrieval gate that bypasses external search for known queries yields an immediate ~85% reduction in latency and operational compute costs.", bold_prefix="2. Compute & Latency Optimization:")
    gen.add_bullet("Context Window & Attention Dilution: Standard RAG pipelines indiscriminately concatenate fixed document windows, causing 'context rot' where irrelevant distractor chunks overwhelm the self-attention mechanism. A utility scoring mechanism is essential to ensure only non-redundant, highly informative passages enter the model cache.", bold_prefix="3. Noise Suppression & Attention Grounding:")
    gen.add_bullet("Hardware Accessibility: Requiring full fine-tuning of 8B+ parameter models creates an insurmountable resource barrier for academic research labs and small-to-medium enterprises. Parameter-efficient LoRA adaptation is crucial to democratize training on single commodity GPUs.", bold_prefix="4. Democratized Parameter-Efficient Training:")

    gen.add_heading_2("1.3 Research Gaps")
    gen.add_p("A rigorous review of contemporary literature reveals six critical research gaps that remain unaddressed in existing retrieval-augmented systems:")
    
    gap_headers = ["Research Gap", "Deficiency in Existing Systems", "Consequence", "Adaptive ImpRAG Solution", "Key References"]
    gap_rows = [
        [
            "Gap 1: Single-Pass Retrieval Rigidity",
            "Existing implicit RAG models execute only a single retrieve-then-generate step.",
            "Complete failure on multi-hop questions (e.g. HotpotQA) where finding evidence requires chaining facts.",
            "Iterative Multi-Hop Retrieval engine with intermediate query refinement and context sufficiency checks.",
            "Lewis et al. (2020), Zhang et al. (2025)"
        ],
        [
            "Gap 2: Prohibitive Training Compute",
            "Baseline ImpRAG mandates full parameter fine-tuning across all transformer layers.",
            "Requires >100 GB VRAM across enterprise clusters, rendering fine-tuning inaccessible.",
            "Native Low-Rank Adaptation (LoRA) on W_Q, W_K in layers 0..b with >99.8% parameter reduction.",
            "Hu et al. (2022), Dettmers et al. (2023)"
        ],
        [
            "Gap 3: Rigid Non-Adaptive Layer Boundaries",
            "Hardcoded layer boundaries (b=7, t=23) applied uniformly across all queries.",
            "Simple queries waste deep middle-layer compute, while complex queries lack sufficient reasoning depth.",
            "Adaptive Layer Boundary Router dynamically selecting Shallow [4,14], Standard [7,20], or Deep [7,26] tiers.",
            "Zhang et al. (2025)"
        ],
        [
            "Gap 4: Absence of Context Utility & Sufficiency",
            "Retrieved nearest neighbors injected directly without relevance verification.",
            "Distractor passages induce attention distraction, conflicting context, and hallucinations.",
            "Document Utility Scorer evaluating semantic relevance U_rel, novelty U_nov, and coverage ratio.",
            "Joren et al. (2025), Asai et al. (2024)"
        ],
        [
            "Gap 5: Uniform Attention Head Averaging",
            "GQA query heads averaged uniformly when computing query embeddings.",
            "Syntactic and noise-carrying heads dilute critical entity-focused semantic representations.",
            "Adaptive GQA Softmax Attention Pooling learning query-conditioned head importance weights alpha_h(q).",
            "Touvron et al. (2023), Zhang et al. (2025)"
        ],
        [
            "Gap 6: Noisy Single-Retriever Supervision",
            "Retriever training relies on noisy pseudo-labels from a single external dense model.",
            "Retriever inherits inductive biases and retrieval errors from the teacher model.",
            "Multi-Retriever Ensemble Supervision fusing dense, BM25 lexical, and ground-truth string inclusion signals.",
            "Zhang et al. (2025)"
        ]
    ]
    gen.add_table(gap_headers, gap_rows, [1400, 2000, 2000, 2200, 1400])

    gen.add_heading_2("1.4 Problem Definition and Scope")
    gen.add_p(
        "Problem Definition: Given an autoregressive language model M with N transformer layers, a dense document corpus C = {p_1, p_2, ..., p_M} indexed in a vector space, and an incoming natural language query q, the objective of Adaptive ImpRAG is to learn an optimal retrieval decision R(q) in {0, 1}, an adaptive passage allocation budget k(q) in {1, 2, 5, 10}, dynamic layer boundaries [b(q), t(q)], and a progressive multi-hop reasoning trajectory that maximizes generation accuracy while minimizing computational latency:"
    )
    gen.add_callout(
        "max_theta E_{q ~ D} [ log P_theta(y | q, C_iter(q)) ] - lambda * ComputeCost(q, k(q), b(q), t(q))\nwhere C_iter(q) represents the utility-filtered, multi-hop consolidated passage cache injected into middle layers b(q)..t(q).",
        title="MATHEMATICAL OPTIMIZATION OBJECTIVE"
    )
    gen.add_p(
        "Scope of the Project:\n"
        "• In-Scope: Open-domain question answering across multi-genre factual datasets (NaturalQuestions, TriviaQA, HotpotQA); dense vector indexing via FAISS with Dual-Centering; implementation on Meta-Llama-3-8B and GPT-2 architectures; parameter-efficient fine-tuning via native LoRA adapters; development of an interactive Gradio web application with real-time decision telemetry.\n"
        "• Out-of-Scope: Real-time dynamic web scraping during live inference (corpus is pre-indexed); multi-modal image/audio retrieval (text-only focus); proprietary closed-source API fine-tuning."
    )

    gen.add_heading_2("1.5 Assumptions and Constraints")
    gen.add_p("The design and execution of Adaptive ImpRAG are governed by the following assumptions and constraints:")
    gen.add_bullet("Corpus Indexing Assumption: It is assumed that the external document corpus (Wikipedia) has been segmented into sentence-bounded passages and pre-encoded into a 4096-dimensional FAISS vector index with Dual-Centering means pre-computed.", bold_prefix="1. Pre-Indexed Knowledge Base:")
    gen.add_bullet("Base Model Weights: It is assumed that pre-trained Meta-Llama-3-8B weights are available in bfloat16 precision and remain frozen except for low-rank adapter modules in layers 0..b.", bold_prefix="2. Frozen LLM Parameters:")
    gen.add_bullet("Hardware Constraints: The prototype is constrained to execute inference within a single NVIDIA A100 (40GB/80GB) or consumer GPU environment, mandating strict VRAM budget management and memory-efficient caching.", bold_prefix="3. Hardware Memory Ceiling:")
    gen.add_bullet("Inference Latency Constraint: The total end-to-end response latency for standard single-hop queries must remain under 300 milliseconds, and under 550 milliseconds for multi-hop iterative queries.", bold_prefix="4. Latency SLA:")

    gen.add_heading_2("1.6 Engineering and Academic Standards")
    gen.add_p("The project adheres rigorously to established international software engineering, quality, and AI ethics standards:")
    gen.add_bullet("IEEE 829-2008 (Standard for Software and System Test Documentation): Applied in designing automated baseline and adaptive test verification suites.", bold_prefix="• Software Testing Standard:")
    gen.add_bullet("ISO/IEC/IEEE 29148-2018 (Systems and Software Engineering — Life Cycle Processes — Requirements Engineering): Guided the Software Requirement Specification (SRS) in Chapter 2.", bold_prefix="• Requirements Standard:")
    gen.add_bullet("ISO/IEC 25010-2011 (Systems and Software Quality Requirements and Evaluation - SQuaRE): Governs performance efficiency, maintainability, and reliability benchmarks.", bold_prefix="• Quality Standard:")
    gen.add_bullet("ISO/IEC 42001-2023 (Artificial Intelligence Management System): Governs ethical AI development, verifiable factual grounding, and mitigation of hallucinated misinformation.", bold_prefix="• AI Ethics Standard:")

    gen.add_heading_2("1.7 Approved Objectives")
    gen.add_p("The project executes six core objectives approved during the Capstone Proposal Evaluation:")
    gen.add_bullet("Objective 1: Analyze limitations of existing RAG frameworks, specifically single-pass rigidity, distractor noise, and prohibitive fine-tuning overhead in baseline ImpRAG.", bold_prefix="Objective 1:")
    gen.add_bullet("Objective 2: Design and implement a 4-dimensional dynamic adaptivity architecture (dynamic gating, dynamic k, dynamic layer slicing, learned GQA pooling).", bold_prefix="Objective 2:")
    gen.add_bullet("Objective 3: Develop an Iterative Multi-Hop Retrieval engine capable of multi-step query decomposition and progressive knowledge refinement.", bold_prefix="Objective 3:")
    gen.add_bullet("Objective 4: Integrate native Parameter-Efficient Fine-Tuning (LoRA) to reduce trainable parameters by >99.8% and democratize training compute.", bold_prefix="Objective 4:")
    gen.add_bullet("Objective 5: Design a Document Utility Scoring and Sentence-Boundary Semantic Chunking pipeline to eliminate noisy and redundant passages.", bold_prefix="Objective 5:")
    gen.add_bullet("Objective 6: Enhance retriever supervision through a Multi-Retriever Ensemble framework combining dense, lexical, and answer grounding signals.", bold_prefix="Objective 6:")

    gen.add_heading_2("1.8 Methodology Overview")
    gen.add_p(
        "The Adaptive ImpRAG methodology organizes the inference pipeline into an integrated 8-stage sequence: (1) Sentence-Boundary Semantic Chunking of raw documents with sliding window overlap; (2) Dual-Centering dense vector indexing; (3) Dynamic query activation extraction at layer b with Learned GQA Head Attention; (4) Dynamic Retrieval Decision Gating for parametric bypass; (5) Entropy-driven dynamic k allocation and layer boundary routing; (6) Iterative multi-hop retrieval loop with context sufficiency checks; (7) Document Utility Scoring and redundancy suppression; and (8) Autoregressive deterministic greedy generation ($T=0.0$) with shifted position IDs."
    )

    gen.add_heading_2("1.9 Project Outcomes and Deliverables")
    gen.add_bullet("Core PyTorch Library (imprag): Modular Python package containing adaptive.py, iterative.py, utility.py, chunking.py, peft_lora.py, model.py, and supervision.py.", bold_prefix="Deliverable 1:")
    gen.add_bullet("Pre-Indexed FAISS Vector Database: 4096-dimensional Dual-Centered index covering structured Wikipedia knowledge.", bold_prefix="Deliverable 2:")
    gen.add_bullet("LoRA Adapter Modules & Checkpoints: Trained low-rank adapter weights for Meta-Llama-3-8B achieving >99.8% parameter efficiency.", bold_prefix="Deliverable 3:")
    gen.add_bullet("Interactive Gradio Web Application: Real-time public interface featuring 2-mode comparison (Adaptive ImpRAG vs Baseline ImpRAG) with live decision telemetry.", bold_prefix="Deliverable 4:")
    gen.add_bullet("Comprehensive Benchmark Evaluation Suite: Automated test scripts (evaluate_retriever.py, test_baseline_verification.py, test_adaptive_verification.py, test_capstone_adaptive_verification.py).", bold_prefix="Deliverable 5:")

    gen.add_heading_2("1.10 Novelty of Adaptive ImpRAG")
    gen.add_p("Adaptive ImpRAG delivers six foundational scientific and algorithmic novelties beyond the state of the art:")
    gen.add_bullet("1. First 4-Dimensional Dynamic Implicit Architecture: Simultaneously unifies dynamic retrieval decisions (when), adaptive passage budgets (how much), dynamic layer depths (where), and learned head pooling (how).", bold_prefix="Novelty 1:")
    gen.add_bullet("2. Multi-Hop Progressive Refinement in Implicit KV Spaces: First implicit RAG architecture to support multi-hop reasoning through iterative intermediate state reformulation.", bold_prefix="Novelty 2:")
    gen.add_bullet("3. Semantic Relevance & Information Novelty Filtering: Re-ranks passages and eliminates redundant distractors prior to middle-layer KV cache injection.", bold_prefix="Novelty 3:")
    gen.add_bullet("4. Sub-0.2% Trainable Parameter Footprint: Low-rank decomposition allows implicit retriever adaptation on a single GPU without touching frozen LLM weights.", bold_prefix="Novelty 4:")
    gen.add_bullet("5. Multi-Retriever Ensemble Pseudo-Labeling: Eliminates teacher retriever inductive bias by fusing dense, lexical, and string-grounding supervision signals.", bold_prefix="Novelty 5:")
    gen.add_bullet("6. Guaranteed Deterministic Greedy Decoding: Integrates explicit EOS stopping and temperature=0.0 greedy token selection to ensure 100% answer reproducibility.", bold_prefix="Novelty 6:")
    gen.add_page_break()

print("Part 1 defined successfully.")
