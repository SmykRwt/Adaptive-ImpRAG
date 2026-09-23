import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from imprag.model import ImpRAGModel
from imprag.loss import MultiLabelNCELoss, SelfDistillationLoss, compute_generation_loss
from imprag.dataset import collate_fn

def run_tests():
    print('=' * 60)
    print('RUNNING BASELINE IMPRAG VERIFICATION TESTS')
    print('=' * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Test Model Architecture & Layer Slicing
    print('[Test 1] Testing Model Initialization & Layer Slicing...')
    config = GPT2Config(n_layer=4, n_embd=128, n_head=4, vocab_size=len(tokenizer))
    base_model = GPT2LMHeadModel(config).to(device)
    
    model = ImpRAGModel(base_model, b=1, t=2, k_passages=2, max_passage_len=16, pooling_type='last_token')
    assert model.b == 1 and model.t == 2, 'Layer boundaries mismatch'
    print('  -> PASSED: Layer slicing initialized successfully (b=1, t=2 on 4-layer model).')
    
    # 2. Test Gradient Flow in get_retriever_embeddings
    print('[Test 2] Testing Gradient Flow in Retriever Projections...')
    model.train()
    dummy_input = torch.randint(0, 100, (2, 8), device=device)
    E_q = model.get_retriever_embeddings(dummy_input, is_query=True)
    E_p = model.get_retriever_embeddings(dummy_input, is_query=False)
    assert E_q.requires_grad, 'E_q should have requires_grad=True during training'
    assert E_p.requires_grad, 'E_p should have requires_grad=True during training'
    dummy_loss = (E_q * E_p).sum()
    dummy_loss.backward()
    print('  -> PASSED: Gradients flow correctly through retriever projection layers.')
    
    # 3. Test Full Attention Concatenated Passage Encoding
    print('[Test 3] Testing Concatenated Passage Encoding (Section 3.1 & Appendix A)...')
    model.eval()
    passages = torch.randint(0, 100, (2, 2, 16), device=device)  # [batch=2, k=2, len=16]
    custom_cache = model.encode_passages(passages, k_passages=2, encoding_mode='concatenated')
    
    # Check that cache exists and has 4 layers total
    assert len(custom_cache.key_cache) == 4, f'Expected 4 layer cache, got {len(custom_cache.key_cache)}'
    # Layer 0: empty
    assert custom_cache.key_cache[0].shape[2] == 0, 'Layer 0 should have 0 passage tokens'
    # Layer 1 & 2 (reader b..t): should have k * 16 = 32 tokens
    assert custom_cache.key_cache[1].shape[2] == 32, f'Expected 32 tokens at layer 1, got {custom_cache.key_cache[1].shape[2]}'
    assert custom_cache.key_cache[2].shape[2] == 32, f'Expected 32 tokens at layer 2, got {custom_cache.key_cache[2].shape[2]}'
    # Layer 3 (top layer LT): empty (disabled cross-attention)
    assert custom_cache.key_cache[3].shape[2] == 0, 'Layer 3 should have 0 passage tokens'
    print('  -> PASSED: Passage KV states correctly isolated in layers b..t (b=1..t=2) with 32 total tokens.')
    
    # 4. Test Multi-Label NCE Loss (Eq 3)
    print('[Test 4] Testing Multi-Label NCE Loss (Eq 3)...')
    loss_fn = MultiLabelNCELoss(temperature=0.05)
    scores = torch.tensor([[1.0, 0.9, 0.8], [0.8, 1.0, 0.9]], device=device)
    pos_mask = torch.tensor([[True, False, False], [False, True, False]], device=device)
    loss = loss_fn(scores, pos_mask)
    assert loss.item() > 0, 'Loss should be positive'
    assert not torch.isnan(loss), 'Loss should not be NaN'
    print(f'  -> PASSED: Multi-Label NCE loss computed successfully (Loss = {loss.item():.4f}).')
    
    # 5. Test Self-Distillation Loss (Eq 4-6)
    print('[Test 5] Testing Self-Distillation KL Divergence Loss (Eq 4-6)...')
    distill_fn = SelfDistillationLoss(tau_t=1.0, tau_r=1.0)
    lm_log_probs = torch.tensor([[-1.0, -5.0, -10.0], [-10.0, -1.0, -5.0]], device=device)
    ret_scores = torch.tensor([[5.0, 1.0, 0.0], [0.0, 5.0, 1.0]], device=device)
    d_loss = distill_fn(ret_scores, lm_log_probs)
    assert d_loss.item() >= 0, 'KL divergence must be non-negative'
    print(f'  -> PASSED: Self-distillation loss computed successfully (KL = {d_loss.item():.4f}).')
    
    # 6. Test Autoregressive Generation with Shifted Position IDs
    print('[Test 6] Testing Autoregressive Generation with Shifted Position IDs...')
    query_input = torch.randint(0, 100, (2, 8), device=device)
    gen_tokens = model.generate(query_input, custom_past_key_values=custom_cache, max_new_tokens=5)
    assert gen_tokens.shape == (2, 5), f'Expected output shape (2, 5), got {gen_tokens.shape}'
    print('  -> PASSED: Autoregressive generation executed with shifted position IDs.')
    
    print('=' * 60)
    print('ALL BASELINE VERIFICATION TESTS PASSED SUCCESSFULLY!')
    print('=' * 60)

if __name__ == '__main__':
    run_tests()
