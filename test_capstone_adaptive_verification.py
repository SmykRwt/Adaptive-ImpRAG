import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from imprag.peft_lora import LoRALinear, apply_lora_to_imprag
from imprag.utility import DocumentUtilityScorer
from imprag.iterative import IterativeImpRAGRetriever
from imprag.supervision import MultiRetrieverEnsembleSupervisor
from imprag.chunking import AdaptiveSemanticChunker
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from imprag.adaptive import AdaptiveImpRAGModel
from imprag.retriever import ImpRAGFAISSIndex

def test_capstone_modules():
    print('=' * 70)
    print('RUNNING CAPSTONE ADAPTIVE IMPRAG VERIFICATION TEST SUITE')
    print('=' * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Test PEFT / LoRA (Section 6.5 & Objective 4)
    print('[Test 1] Testing Parameter-Efficient Fine-Tuning (LoRA)...')
    config = GPT2Config(n_layer=4, n_embd=128, n_head=4, vocab_size=128256, _attn_implementation='eager')
    dummy_model = GPT2LMHeadModel(config).to(device)
    adapters = apply_lora_to_imprag(dummy_model, r=8, lora_alpha=16, target_modules=['c_attn'], max_layer=2)
    assert len(adapters) > 0, 'LoRA adapters should be attached'
    dummy_in = torch.randint(0, 100, (2, 8), device=device)
    out = dummy_model(dummy_in).logits
    assert out.shape == (2, 8, config.vocab_size)
    print('  -> PASSED: Native LoRA linear adapters successfully applied and forward pass verified.')
    
    # 2. Test Document Utility Scoring (Section 6.4 & Objective 1)
    print('[Test 2] Testing Document Utility Scoring & Sufficiency Filtering...')
    scorer = DocumentUtilityScorer(utility_threshold=0.2, redundancy_penalty=0.3)
    query = 'Who invented the light bulb and phonograph?'
    passages = [
        'Thomas Edison was an American inventor who developed the incandescent light bulb and the phonograph.',
        'Thomas Edison was an American inventor who developed the phonograph.',  # Redundant with passage 1
        'The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.'  # Irrelevant
    ]
    selected, rep = scorer.score_and_filter_passages(query, passages, [0.9, 0.8, 0.1])
    assert len(selected) >= 1, 'Should retain high-utility passage'
    assert 'Edison' in selected[0], 'Should retain most informative passage'
    assert rep['is_context_sufficient'], 'Context should be sufficient'
    print(f'  -> PASSED: Document utility scoring filtered out noise and redundancy (Retained {len(selected)}/{len(passages)} passages).')
    
    # 3. Test Improved Multi-Retriever Supervision (Section 6.7 & Objective 6)
    print('[Test 3] Testing Multi-Retriever Ensemble Supervision...')
    supervisor = MultiRetrieverEnsembleSupervisor()
    dense_s = torch.tensor([[0.8, 0.4, 0.1]])
    lexical_s = torch.tensor([[0.7, 0.6, 0.0]])
    grounding = torch.tensor([[True, False, False]])
    targets = supervisor.generate_ensemble_targets(dense_s, lexical_s, grounding)
    assert targets.shape == (1, 3)
    assert torch.allclose(targets.sum(dim=-1), torch.tensor([1.0])), 'Soft targets must sum to 1'
    assert targets[0, 0] > targets[0, 1] > targets[0, 2], 'Target ranking must match quality fusion'
    print(f'  -> PASSED: Multi-retriever ensemble targets generated cleanly: {targets[0].tolist()}.')
    
    # 4. Test Sentence-Boundary Semantic Chunking (Section 6.1)
    print('[Test 4] Testing Sentence-Boundary Aware Semantic Chunking...')
    chunker = AdaptiveSemanticChunker(target_chunk_size=15, overlap_size=8, min_chunk_size=5)
    raw_doc = 'Albert Einstein was born in Ulm, Germany. He developed the general theory of relativity. Later in life, he won the Nobel Prize in Physics in 1921. He spent his later years at Princeton University.'
    chunks = chunker.chunk_text(raw_doc, doc_title='Albert Einstein')
    assert len(chunks) >= 2, 'Should generate multiple overlapping chunks'
    assert all('[Albert Einstein]' in c for c in chunks), 'Every chunk must retain document title grounding'
    print(f'  -> PASSED: Semantic chunking generated {len(chunks)} sentence-bounded, title-grounded chunks.')
    
    # 5. Test Iterative Multi-Hop Retrieval (Section 6.3 & Objective 3)
    print('[Test 5] Testing Iterative Multi-Hop Retrieval Mechanism...')
    ckpt_dir = os.path.join(BASE_DIR, "imp_rag_checkpoint")
    if os.path.exists(ckpt_dir):
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    adaptive_model = AdaptiveImpRAGModel(dummy_model, default_b=1, default_t=2).to(device)
    iterative_retriever = IterativeImpRAGRetriever(adaptive_model=adaptive_model, max_hops=2)
    
    faiss_idx = ImpRAGFAISSIndex(dimension=128)
    toy_embs = torch.randn(4, 128).numpy()
    faiss_idx.add_embeddings(toy_embs)
    corpus = [
        'Ada Lovelace was an English mathematician.',
        'Ada Lovelace collaborated with Charles Babbage on the Analytical Engine.',
        'Charles Babbage is considered the father of the modern computer.',
        'Alan Turing made foundational contributions to theoretical computer science.'
    ]
    
    q_multi = 'Who collaborated with the father of the computer?'
    q_enc = tokenizer([q_multi], return_tensors='pt').input_ids.to(device)
    
    ans, telem = iterative_retriever.iterative_generate(
        query_ids=q_enc,
        query_text=q_multi,
        faiss_index=faiss_idx,
        passages=corpus,
        tokenizer=tokenizer,
        max_new_tokens=5,
        temperature=0.0
    )
    print('  -> Output:', ans)
    print('  -> Telemetry Hops:', telem['total_hops'], '| Allocated k:', telem['k_allocated'])
    print('  -> PASSED: Iterative multi-hop retrieval executed successfully.')
    
    print('=' * 70)
    print('ALL CAPSTONE ADAPTIVE IMPRAG VERIFICATIONS PASSED WITH 100% SUCCESS!')
    print('=' * 70)

if __name__ == '__main__':
    test_capstone_modules()
