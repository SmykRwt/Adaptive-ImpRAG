import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

def extract_kv_for_layer(passage_cache, l):
    """
    Extracts key and value tensors for layer l from passage_cache,
    handling DynamicCache (various versions) and legacy tuple structures.
    """
    if isinstance(passage_cache, tuple):
        layer_obj = passage_cache[l]
        if isinstance(layer_obj, (tuple, list)):
            return layer_obj[0], layer_obj[1]
        elif hasattr(layer_obj, "keys") and hasattr(layer_obj, "values"):
            return layer_obj.keys, layer_obj.values
    elif hasattr(passage_cache, "key_cache") and hasattr(passage_cache, "value_cache") and len(passage_cache.key_cache) > l:
        return passage_cache.key_cache[l], passage_cache.value_cache[l]
    elif hasattr(passage_cache, "layers") and len(passage_cache.layers) > l:
        layer_obj = passage_cache.layers[l]
        if hasattr(layer_obj, "keys") and hasattr(layer_obj, "values"):
            return layer_obj.keys, layer_obj.values
        elif isinstance(layer_obj, (tuple, list)):
            return layer_obj[0], layer_obj[1]
    
    # Fallback indexing
    layer_obj = passage_cache[l]
    if isinstance(layer_obj, (tuple, list)):
        return layer_obj[0], layer_obj[1]
    return getattr(layer_obj, "keys", layer_obj[0]), getattr(layer_obj, "values", layer_obj[1])

def safe_update_dynamic_cache(cache, k, v, layer_idx):
    """
    Safely updates DynamicCache for arbitrary layer_idx by pre-filling 0-length
    tensors for preceding layers if cache.key_cache is shorter than layer_idx.
    """
    num_heads = k.shape[1]
    head_dim = k.shape[3]
    batch_size = k.shape[0]
    
    while len(cache.key_cache) < layer_idx:
        dummy_k = torch.empty((batch_size, num_heads, 0, head_dim), device=k.device, dtype=k.dtype)
        dummy_v = torch.empty((batch_size, num_heads, 0, head_dim), device=v.device, dtype=v.dtype)
        cache.key_cache.append(dummy_k)
        cache.value_cache.append(dummy_v)
        
    cache.update(k, v, layer_idx)

def pad_cache_to_num_layers(cache, num_layers, batch_size, num_heads, head_dim, device, dtype):
    """
    Fills trailing layers (t+1..num_layers-1) with 0-length tensors so cache length matches total model layers.
    """
    while len(cache.key_cache) < num_layers:
        dummy_k = torch.empty((batch_size, num_heads, 0, head_dim), device=device, dtype=dtype)
        dummy_v = torch.empty((batch_size, num_heads, 0, head_dim), device=device, dtype=dtype)
        cache.key_cache.append(dummy_k)
        cache.value_cache.append(dummy_v)

class ImpRAGModel(nn.Module):
    """
    Wrapper around an autoregressive CausalLM model supporting the ImpRAG architecture (Section 3.1):
    1. Layer Slicing:
       - Bottom group L_B (layers 0..b): Acts as retriever (Section 3.1). Default b=7 (Llama-3-8B / 3B).
       - Middle group L_M (layers b..t): Acts as implicit reader/cache. Default t=23 (8B) / t=19 (3B).
       - Top group L_T (layers t+1..N-1): Acts as generator with cross-attention to passages disabled.
    2. Retriever Projections & GQA Group-Averaging:
       - Extracts query Q and key K states at layer b.
       - Pools last token (or active tokens mean) to get E_q^g in R^{(h_k * g) * d_h}.
       - Averages heads within each of the h_k groups -> E_q in R^{h_k * d_h}.
       - Extracts last token K state -> E_p in R^{h_k * d_h}.
       - Inner product similarity: s(q, p) = E_q . E_p (Eq 1).
    3. Reader Passage Encoding:
       - Concatenated Encoding (Full Attention, Appendix A & Section 3.1): Jointly encodes all k
         retrieved passages concatenated together to allow inter-passage cross-attention in L_M.
       - Injects passage KV states into DynamicCache strictly inside reader layers b..t.
    4. Shifted Position IDs:
       - Shifts query/response token position IDs by total passage length to avoid RoPE interference.
    """
    def __init__(self, base_model, b=7, t=23, k_passages=5, max_passage_len=128, pooling_type="last_token"):
        super().__init__()
        self.base_model = base_model
        self.b = b
        self.t = t
        self.k_passages = k_passages
        self.max_passage_len = max_passage_len
        self.pooling_type = pooling_type
        
        # Determine total layers and adjust boundaries if needed
        num_layers = self._get_num_layers()
        if self.b >= num_layers:
            self.b = max(0, num_layers // 2 - 1)
        if self.t >= num_layers:
            self.t = max(self.b + 1, int(num_layers * 0.75))
            
        # Detect model architecture type and locate projection modules at layer b
        self.model_type, self.q_module, self.k_module = self._find_projection_modules(self.b)
        
        # Activations captured during forward pass
        self.captured_q = None
        self.captured_k = None
        
        # Register hooks
        self._register_hooks()

    def _get_num_layers(self):
        if hasattr(self.base_model, "config"):
            if hasattr(self.base_model.config, "num_hidden_layers"):
                return self.base_model.config.num_hidden_layers
            if hasattr(self.base_model.config, "n_layer"):
                return self.base_model.config.n_layer
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            return len(self.base_model.model.layers)
        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "h"):
            return len(self.base_model.transformer.h)
        return 32

    def _get_layers_list(self):
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            return self.base_model.model.layers
        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "h"):
            return self.base_model.transformer.h
        raise AttributeError("Could not identify transformer layers in base_model.")

    def _find_projection_modules(self, b):
        """
        Dynamically finds the query and key projection modules of layer b based on the model type.
        """
        # LLaMA / Mistral / Qwen style
        try:
            layers = self._get_layers_list()
            layer = layers[b]
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "q_proj") and hasattr(layer.self_attn, "k_proj"):
                    return "llama", layer.self_attn.q_proj, layer.self_attn.k_proj
        except Exception:
            pass
        
        # GPT-2 style
        try:
            if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "h"):
                layer = self.base_model.transformer.h[b]
                if hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
                    return "gpt2", layer.attn.c_attn, None
        except Exception:
            pass
            
        raise ValueError(f"Unsupported model architecture or layer index b={b} is out of range.")

    def _register_hooks(self):
        """
        Registers forward hooks on the query and key projection modules at layer b.
        No detach() is used so that gradients flow back during retriever training.
        """
        if self.model_type == "llama":
            def q_hook(module, input, output):
                self.captured_q = output
            def k_hook(module, input, output):
                self.captured_k = output
            self.q_hook_handle = self.q_module.register_forward_hook(q_hook)
            self.k_hook_handle = self.k_module.register_forward_hook(k_hook)
            
        elif self.model_type == "gpt2":
            def gpt2_hook(module, input, output):
                split_size = output.shape[-1] // 3
                q, k, v = output.split(split_size, dim=-1)
                self.captured_q = q
                self.captured_k = k
            self.q_hook_handle = self.q_module.register_forward_hook(gpt2_hook)
            self.k_hook_handle = None

    def clear_captured_states(self):
        """
        Clears stored activations to free GPU memory.
        """
        self.captured_q = None
        self.captured_k = None

    def get_retriever_embeddings(self, input_ids, attention_mask=None, is_query=True):
        """
        Computes retriever embeddings using bottom layer group L_B (layers 0..b) according to Section 3.1:
        - Executes forward pass through layers 0..b to trigger hook at layer b.
        - Pools last active token (or active token mean).
        - GQA group-averages query heads across each of the h_k groups -> E_q in R^{h_k * d_h}.
        - Extracts key projection -> E_p in R^{h_k * d_h}.
        - Allows gradient propagation when training.
        """
        self.clear_captured_states()
        
        layers = self._get_layers_list()
        orig_layers = list(layers)
        
        # Temporarily slice layer list to 0..b
        if self.model_type == "llama":
            self.base_model.model.layers = nn.ModuleList(orig_layers[:self.b + 1])
        elif self.model_type == "gpt2":
            self.base_model.transformer.h = nn.ModuleList(orig_layers[:self.b + 1])
            
        try:
            self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False
            )
        finally:
            if self.model_type == "llama":
                self.base_model.model.layers = nn.ModuleList(orig_layers)
            elif self.model_type == "gpt2":
                self.base_model.transformer.h = nn.ModuleList(orig_layers)
                
        captured_q = self.captured_q
        captured_k = self.captured_k
        self.clear_captured_states()
        
        if captured_q is None or captured_k is None:
            raise RuntimeError("Hook failed to capture activations at layer b. Check layer configuration.")
            
        batch_size = input_ids.shape[0]
        
        # Token pooling over sequence
        if self.pooling_type == "last_token":
            if attention_mask is not None:
                last_indices = (attention_mask.sum(dim=-1) - 1).clamp(min=0)
            else:
                last_indices = torch.full((batch_size,), input_ids.shape[1] - 1, dtype=torch.long, device=input_ids.device)
                
            q_pooled = torch.stack([captured_q[i, last_indices[i]] for i in range(batch_size)])
            k_pooled = torch.stack([captured_k[i, last_indices[i]] for i in range(batch_size)])
        else:
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                q_pooled = (captured_q * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                k_pooled = (captured_k * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                q_pooled = captured_q.mean(dim=1)
                k_pooled = captured_k.mean(dim=1)
                
        # Grouped-Query Attention (GQA) group-averaging (Section 3.1)
        if self.model_type == "llama":
            num_attn_heads = self.base_model.config.num_attention_heads
            num_kv_heads = self.base_model.config.num_key_value_heads
            head_dim = self.base_model.config.hidden_size // num_attn_heads
            g = num_attn_heads // num_kv_heads
            
            q_reshaped = q_pooled.view(batch_size, num_kv_heads, g, head_dim)
            E_q = q_reshaped.mean(dim=2).view(batch_size, num_kv_heads * head_dim)
            E_p = k_pooled.view(batch_size, num_kv_heads * head_dim)
        else:
            E_q = q_pooled.view(batch_size, -1)
            E_p = k_pooled.view(batch_size, -1)
            
        return E_q if is_query else E_p

    def encode_passages(self, passage_ids, attention_mask=None, k_passages=None, encoding_mode="concatenated"):
        """
        Encodes retrieved passages to generate KV states for middle layer group L_M (layers b..t).
        """
        was_training = self.base_model.training
        self.base_model.eval()
        
        num_layers = self._get_num_layers()
        custom_cache = DynamicCache()
        sample_k_state = None
        
        try:
            if encoding_mode == "concatenated":
                if passage_ids.dim() == 3:
                    batch_size, k, seq_len = passage_ids.shape
                    concat_ids = passage_ids.view(batch_size, k * seq_len)
                    concat_mask = attention_mask.view(batch_size, k * seq_len) if attention_mask is not None else None
                elif passage_ids.dim() == 2:
                    k = k_passages if k_passages is not None else self.k_passages
                    if passage_ids.shape[0] % k == 0 and passage_ids.shape[0] > 1 and k > 1:
                        batch_size = passage_ids.shape[0] // k
                        seq_len = passage_ids.shape[1]
                        concat_ids = passage_ids.view(batch_size, k * seq_len)
                        concat_mask = attention_mask.view(batch_size, k * seq_len) if attention_mask is not None else None
                    else:
                        batch_size = passage_ids.shape[0]
                        concat_ids = passage_ids
                        concat_mask = attention_mask
                else:
                    raise ValueError(f"Unexpected passage_ids shape: {passage_ids.shape}")
                    
                with torch.no_grad():
                    outputs = self.base_model(
                        input_ids=concat_ids,
                        attention_mask=concat_mask,
                        use_cache=True
                    )
                    
                for l in range(num_layers):
                    if self.b <= l <= self.t:
                        k_state, v_state = extract_kv_for_layer(outputs.past_key_values, l)
                        safe_update_dynamic_cache(custom_cache, k_state, v_state, l)
                        sample_k_state = k_state
                        
            else:
                batch_k, passage_len = passage_ids.shape
                k = k_passages if k_passages is not None else (self.k_passages if batch_k % self.k_passages == 0 else 1)
                batch_size = batch_k // k
                
                with torch.no_grad():
                    outputs = self.base_model(
                        input_ids=passage_ids,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
                    
                for l in range(num_layers):
                    if self.b <= l <= self.t:
                        k_state, v_state = extract_kv_for_layer(outputs.past_key_values, l)
                        num_heads, _, head_dim = k_state.shape[1], k_state.shape[2], k_state.shape[3]
                        
                        k_state = k_state.view(batch_size, k, num_heads, passage_len, head_dim).permute(0, 2, 1, 3, 4).reshape(batch_size, num_heads, k * passage_len, head_dim)
                        v_state = v_state.view(batch_size, k, num_heads, passage_len, head_dim).permute(0, 2, 1, 3, 4).reshape(batch_size, num_heads, k * passage_len, head_dim)
                        safe_update_dynamic_cache(custom_cache, k_state, v_state, l)
                        sample_k_state = k_state
                        
            # Pad cache to num_layers to prevent IndexError on top layers (t+1..N-1)
            if sample_k_state is not None:
                pad_cache_to_num_layers(
                    custom_cache,
                    num_layers=num_layers,
                    batch_size=sample_k_state.shape[0],
                    num_heads=sample_k_state.shape[1],
                    head_dim=sample_k_state.shape[3],
                    device=sample_k_state.device,
                    dtype=sample_k_state.dtype
                )
        finally:
            self.base_model.train(was_training)
            
        return custom_cache

    def forward(self, query_ids, custom_past_key_values=None, labels=None, attention_mask=None):
        """
        Runs query through the model with pre-encoded passage KV states cross-attended in layers b..t.
        Shifts query position IDs by passage length (Section 3.1).
        """
        batch_size, query_len = query_ids.shape
        
        k_max_len = 0
        if custom_past_key_values is not None and hasattr(custom_past_key_values, "key_cache"):
            if len(custom_past_key_values.key_cache) > self.b and custom_past_key_values.key_cache[self.b] is not None:
                if custom_past_key_values.key_cache[self.b].ndim >= 3:
                    k_max_len = custom_past_key_values.key_cache[self.b].shape[2]
                    
        position_ids = torch.arange(k_max_len, k_max_len + query_len, device=query_ids.device).unsqueeze(0).repeat(batch_size, 1)
        
        return self.base_model(
            input_ids=query_ids,
            past_key_values=custom_past_key_values,
            position_ids=position_ids,
            labels=labels,
            attention_mask=attention_mask if custom_past_key_values is None else None
        )

    def generate(self, query_ids, custom_past_key_values=None, max_new_tokens=30, temperature=0.0, eos_token_ids=None):
        """
        Autoregressive generation loop with passage KV states prepended and shifted position IDs.
        Defaults to deterministic greedy decoding (temperature=0.0) with EOS early stopping.
        """
        self.eval()
        with torch.no_grad():
            batch_size, query_len = query_ids.shape
            
            if eos_token_ids is None:
                eos_token_ids = {self.base_model.config.eos_token_id}
            elif isinstance(eos_token_ids, int):
                eos_token_ids = {eos_token_ids}
            else:
                eos_token_ids = set(eos_token_ids)
                
            k_max_len = 0
            if custom_past_key_values is not None and hasattr(custom_past_key_values, "key_cache"):
                if len(custom_past_key_values.key_cache) > self.b and custom_past_key_values.key_cache[self.b] is not None:
                    if custom_past_key_values.key_cache[self.b].ndim >= 3:
                        k_max_len = custom_past_key_values.key_cache[self.b].shape[2]
                        
            position_ids = torch.arange(k_max_len, k_max_len + query_len, device=query_ids.device).unsqueeze(0).repeat(batch_size, 1)
            
            outputs = self.base_model(
                input_ids=query_ids,
                past_key_values=custom_past_key_values,
                position_ids=position_ids,
                attention_mask=None
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            if temperature > 0.0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            generated = [next_tokens]
            past_key_values = outputs.past_key_values
            if isinstance(past_key_values, tuple) and hasattr(DynamicCache, "from_legacy_cache"):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                
            # Check if first token is EOS (for single-item batches)
            if batch_size == 1 and next_tokens.item() in eos_token_ids:
                return torch.cat(generated, dim=-1)
                
            for i in range(max_new_tokens - 1):
                next_pos = k_max_len + query_len + i
                pos_tensor = torch.tensor([[next_pos]], device=query_ids.device).repeat(batch_size, 1)
                
                outputs = self.base_model(
                    input_ids=next_tokens,
                    past_key_values=past_key_values,
                    position_ids=pos_tensor,
                    attention_mask=None
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                if temperature > 0.0:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                generated.append(next_tokens)
                past_key_values = outputs.past_key_values
                if isinstance(past_key_values, tuple) and hasattr(DynamicCache, "from_legacy_cache"):
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                    
                if batch_size == 1 and next_tokens.item() in eos_token_ids:
                    break
                    
            return torch.cat(generated, dim=-1)
