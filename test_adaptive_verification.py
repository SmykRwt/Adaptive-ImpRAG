import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from imprag.retriever import ImpRAGFAISSIndex
from imprag.adaptive import AdaptiveImpRAGModel, AdaptiveGQAPooling, DynamicRetrievalGate, AdaptiveKAllocator, AdaptiveLayerBoundaryRouter

def run_adaptive_tests():
    print('=' * 60)
    print('RUNNING ADAPTIVE IMPRAG 4-DIMENSIONAL VERIFICATION TESTS')
    print('=' * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = os.path.join(BASE_DIR, "imp_rag_checkpoint")
    if os.path.exists(ckpt_dir):
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Test Dimension 4: Adaptive GQA Head Pooling
    print('[Dimension 4 Test] Testing Adaptive GQA Head Pooling...')
    pooling = AdaptiveGQAPooling(num_kv_heads=8, g=4, head_dim=128).to(device)
    q_dummy = torch.randn(2, 8, 4, 128, device=device, requires_grad=True)
    E_q, attn_weights = pooling(q_dummy)
    assert E_q.shape == (2, 1024), f'Expected shape (2, 1024), got {E_q.shape}'
    assert attn_weights.shape == (2, 8, 4), f'Expected shape (2, 8, 4), got {attn_weights.shape}'
    # Test sum to 1
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(2, 8, device=device)), 'Head weights must sum to 1'
    loss = E_q.sum()
    loss.backward()
    assert pooling.head_scores.grad is not None, 'Gradients must flow to head_scores'
    print('  -> PASSED: Dynamic learned head attention weighting works with proper gradients and normalization.')
    
    # 2. Test Dimension 1: Dynamic Retrieval Decision (Parametric Bypass Gate)
    print('[Dimension 1 Test] Testing Dynamic Retrieval Gate...')
    gate = DynamicRetrievalGate(input_dim=1024, threshold=0.5).to(device)
    dummy_emb = torch.randn(4, 1024, device=device)
    probs, should_ret = gate(dummy_emb)
    assert probs.shape == (4,), f'Expected shape (4,), got {probs.shape}'
    assert should_ret.shape == (4,), f'Expected shape (4,), got {should_ret.shape}'
    print(f'  -> PASSED: Retrieval decision evaluated (Sample probs: {probs[:2].tolist()}).')
    
    # 3. Test Dimension 2: Dynamic k Allocation
    print('[Dimension 2 Test] Testing Dynamic k Allocation...')
    allocator = AdaptiveKAllocator(tau=0.1)
    
    # Case A: Clear single dominant passage (high margin) -> expect k=1
    scores_dominant = np.array([10.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1])
    k_dom, meta_dom = allocator.compute_k(scores_dominant)
    assert k_dom == 1, f'Expected k=1 for dominant score, got {k_dom}'
    print(f'  -> PASSED Case A (Dominant Document): Selected k={k_dom} (Margin={meta_dom["margin"]:.2f}, Entropy={meta_dom["entropy"]:.2f}).')
    
    # Case B: Diffuse / high entropy scores (multi-hop) -> expect k=5 or k=10
    scores_diffuse = np.array([2.05, 2.04, 2.03, 2.02, 2.01, 2.00, 1.99, 1.98, 1.97, 1.96])
    k_diff, meta_diff = allocator.compute_k(scores_diffuse)
    assert k_diff in [5, 10], f'Expected k=5 or 10 for diffuse scores, got {k_diff}'
    print(f'  -> PASSED Case B (Diffuse Multi-Fact): Selected k={k_diff} (Margin={meta_diff["margin"]:.2f}, Entropy={meta_diff["entropy"]:.2f}).')
    
    # 4. Test Dimension 3: Adaptive Layer Slicing Router
    print('[Dimension 3 Test] Testing Adaptive Layer Slicing [b(q), t(q)] Router...')
    router = AdaptiveLayerBoundaryRouter(input_dim=1024, num_layers=32).to(device)
    tier_probs, tier_indices = router(dummy_emb)
    assert tier_probs.shape == (4, 3), f'Expected tier_probs shape (4, 3), got {tier_probs.shape}'
    selected_tier = router.tiers[tier_indices[0].item()]
    print(f'  -> PASSED: Boundary tier selected: {selected_tier["name"]} -> [b={selected_tier["b"]}, t={selected_tier["t"]}].')
    
    # 5. Test Full AdaptiveImpRAGModel Inference Pipeline
    print('[End-to-End Test] Testing Full Adaptive ImpRAG Pipeline on Toy Corpus...')
    config = GPT2Config(n_layer=4, n_embd=128, n_head=4, vocab_size=len(tokenizer))
    base_model = GPT2LMHeadModel(config).to(device)
    
    adaptive_model = AdaptiveImpRAGModel(base_model, default_b=1, default_t=2).to(device)
    
    # Build toy FAISS index
    toy_passages = [
        'Paris is the capital of France, famous for the Eiffel Tower.',
        'Shakespeare was an English poet and playwright who wrote Hamlet.',
        'The Pacific Ocean is the deepest ocean on planet Earth.'
    ]
    
    faiss_idx = ImpRAGFAISSIndex(dimension=128)
    toy_embs = np.random.randn(3, 128).astype(np.float32)
    faiss_idx.add_embeddings(toy_embs)
    
    query = 'Who wrote Hamlet?'
    q_enc = tokenizer([query], return_tensors='pt').input_ids.to(device)
    
    # Test with Force Retrieve = True
    gen_text, telem = adaptive_model.adaptive_generate(
        query_ids=q_enc,
        query_text=query,
        faiss_index=faiss_idx,
        passages=toy_passages,
        tokenizer=tokenizer,
        max_new_tokens=5,
        force_retrieve=True
    )
    
    print('  -> Output with Retrieval:', gen_text)
    print('  -> Telemetry Summary:', {
        'Decision': telem['retrieval_decision'],
        'Allocated k': telem['k_allocated'],
        'Boundaries': telem['layer_boundaries'],
        'Compute Saved': telem['compute_saved']
    })
    
    # Test with Force Retrieve = False (Parametric Bypass)
    gen_text_bypass, telem_bypass = adaptive_model.adaptive_generate(
        query_ids=q_enc,
        query_text=query,
        faiss_index=faiss_idx,
        passages=toy_passages,
        tokenizer=tokenizer,
        max_new_tokens=5,
        force_retrieve=False
    )
    assert telem_bypass['retrieval_decision'] == 'PARAMETRIC_BYPASS', 'Expected PARAMETRIC_BYPASS'
    assert telem_bypass['k_allocated'] == 0, 'Bypassed query should have k=0'
    print('  -> Output with Parametric Bypass:', gen_text_bypass)
    print('  -> Telemetry Bypass:', telem_bypass['compute_saved'])
    
    print('=' * 60)
    print('ALL ADAPTIVE IMPRAG 4-DIMENSION TESTS PASSED PERFECTLY!')
    print('=' * 60)

if __name__ == '__main__':
    run_adaptive_tests()
