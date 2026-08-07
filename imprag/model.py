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
    
    # Pre-fill preceding layers with 0-seq-len dummy tensors to prevent IndexError in transformers DynamicCache
    while len(cache.key_cache) < layer_idx:
        dummy_k = torch.empty((batch_size, num_heads, 0, head_dim), device=k.device, dtype=k.dtype)
        dummy_v = torch.empty((batch_size, num_heads, 0, head_dim), device=v.device, dtype=v.dtype)
        cache.key_cache.append(dummy_k)
        cache.value_cache.append(dummy_v)
        
    cache.update(k, v, layer_idx)

class ImpRAGModel(nn.Module):
    """
    Wrapper around a Hugging Face CausalLM model to support the ImpRAG architecture.
    Handles:
    1. Layer Slicing: partition layers into bottom group L_B (0..b), middle L_M (b..t), top L_T (t+1..N-1).
    2. Bottom layers as retriever: extracts Q/K states at layer b, performs Grouped-Query Attention pooling.
    3. Middle layers as reader: prepends retrieved passage KV states inside layers b..t using DynamicCache.
    4. Shifted Position IDs: shifts query/response token position IDs to avoid interference with passage RoPE.
    """
    def __init__(self, base_model, b, t, k_passages=5, max_passage_len=128, pooling_type="mean"):
        super().__init__()
        self.base_model = base_model
        self.b = b
        self.t = t
        self.k_passages = k_passages
        self.max_passage_len = max_passage_len
        self.pooling_type = pooling_type
        
        # Detect model type and find the target projection modules at layer b
        self.model_type, self.q_module, self.k_module = self._find_projection_modules(b)
        
        # Activations captured during forward pass
        self.captured_q = None
        self.captured_k = None
        
        # Register PyTorch hooks to capture projections
        self._register_hooks()

    def _find_projection_modules(self, b):
        """
        Dynamically finds the query and key projection modules of layer b based on the model type.
        """
        # Try Llama path
        try:
            if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
                layer = self.base_model.model.layers[b]
                if hasattr(layer.self_attn, 'q_proj') and hasattr(layer.self_attn, 'k_proj'):
                    return "llama", layer.self_attn.q_proj, layer.self_attn.k_proj
        except Exception:
            pass
        
        # Try GPT-2 path
        try:
            if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
                layer = self.base_model.transformer.h[b]
                if hasattr(layer.attn, 'c_attn'):
                    return "gpt2", layer.attn.c_attn, None
        except Exception:
            pass
            
        raise ValueError(f"Unsupported model architecture or layer index b={b} is out of range.")

    def _register_hooks(self):
        """
        Registers forward hooks on the query and key projection modules at layer b.
        No detach() is used in the hook function so that gradients can flow back
        during retriever training.
        """
        if self.model_type == "llama":
            def q_hook(module, input, output):
                self.captured_q = output
            def k_hook(module, input, output):
                self.captured_k = output
            self.q_module.register_forward_hook(q_hook)
            self.k_module.register_forward_hook(k_hook)
            
        elif self.model_type == "gpt2":
            def gpt2_hook(module, input, output):
                # GPT-2 c_attn projects Q, K, V concatenated [batch, seq, 3 * embed_dim]
                split_size = output.shape[-1] // 3
                q, k, v = output.split(split_size, dim=2)
                self.captured_q = q
                self.captured_k = k
            self.q_module.register_forward_hook(gpt2_hook)

    def clear_captured_states(self):
        """
        Clears the stored query and key states to prevent memory leaks.
        """
        self.captured_q = None
        self.captured_k = None

    def get_retriever_embeddings(self, input_ids, attention_mask=None, is_query=True):
        """
        Computes the retriever embedding for the input sequence.
        Supports:
        - "last_token": Paper standard last-token pooling.
        - "mean": Mean pooling over active tokens.
        """
        self.clear_captured_states()
        
        # Run forward pass through the bottom layers up to layer b to trigger the hooks
        original_layers = self.base_model.model.layers
        self.base_model.model.layers = original_layers[:self.b + 1]
        try:
            with torch.no_grad():
                self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                )
        finally:
            self.base_model.model.layers = original_layers
            
        captured_q = self.captured_q
        captured_k = self.captured_k
        self.clear_captured_states()
        
        if captured_q is None or (captured_k is None and self.model_type == "llama"):
            raise RuntimeError("Hook failed to capture activations. Check if forward pass ran correctly.")
            
        batch_size = input_ids.shape[0]
        
        if self.pooling_type == "last_token":
            # Find last active token index for each item in the batch
            if attention_mask is not None:
                last_indices = attention_mask.sum(dim=-1) - 1
            else:
                last_indices = torch.full((batch_size,), input_ids.shape[1] - 1, dtype=torch.long, device=input_ids.device)
                
            q_pooled = torch.stack([captured_q[i, last_indices[i]] for i in range(batch_size)])
            k_pooled = torch.stack([captured_k[i, last_indices[i]] for i in range(batch_size)])
        else:
            # Mean pooling over query/passage tokens
            if attention_mask is not None:
                mask = torch.zeros_like(attention_mask, dtype=torch.float, device=input_ids.device)
                for i in range(batch_size):
                    last_idx = int(attention_mask[i].sum().item()) - 1
                    if is_query:
                        if last_idx > 4:
                            mask[i, 2 : last_idx - 1] = 1.0
                        else:
                            mask[i, : last_idx + 1] = 1.0
                    else:
                        mask[i, : last_idx + 1] = 1.0
            else:
                mask = torch.zeros(batch_size, input_ids.shape[1], dtype=torch.float, device=input_ids.device)
                if is_query:
                    if input_ids.shape[1] > 4:
                        mask[:, 2 : input_ids.shape[1] - 2] = 1.0
                    else:
                        mask[:, :] = 1.0
                else:
                    mask[:, :] = 1.0
                    
            mask_expanded = mask.unsqueeze(-1)
            
            # Mean pool Query projections
            q_masked = captured_q * mask_expanded
            q_pooled = q_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            
            # Mean pool Key projections
            k_masked = captured_k * mask_expanded
            k_pooled = k_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            
        # Grouped-Query Attention head averaging
        num_attn_heads = self.base_model.config.num_attention_heads
        num_kv_heads = self.base_model.config.num_key_value_heads
        head_dim = self.base_model.config.hidden_size // num_attn_heads
        g = num_attn_heads // num_kv_heads
        
        # Reshape and average query heads
        q_reshaped = q_pooled.view(batch_size, num_kv_heads, g, head_dim)
        E_q = q_reshaped.mean(dim=2).view(batch_size, -1)
        
        # Flatten key projection
        E_p = k_pooled.view(batch_size, -1)
        
        return E_q if is_query else E_p

    def encode_passages(self, passage_ids, attention_mask=None, k_passages=None):
        """
        Runs the passages through the bottom and middle layers to get their KV states.
        passage_ids: [batch_size * k, passage_len]
        Returns:
            DynamicCache containing the passage KV states reshaped to [batch_size, num_heads, k * passage_len, head_dim]
            for reader layers b..t, and empty for other layers.
        """
        batch_k, passage_len = passage_ids.shape
        k = k_passages if k_passages is not None else (self.k_passages if batch_k % self.k_passages == 0 else 1)
        batch_size = batch_k // k
        
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            # Sub-chunk passage forward pass if total passages > 16 to avoid NVML VRAM peaks on MIG partitions
            sub_batch_size = 16
            if batch_k > sub_batch_size:
                all_k_states = {l: [] for l in range(self.b, self.t + 1)}
                all_v_states = {l: [] for l in range(self.b, self.t + 1)}
                
                for i in range(0, batch_k, sub_batch_size):
                    sub_ids = passage_ids[i : i + sub_batch_size]
                    sub_mask = attention_mask[i : i + sub_batch_size] if attention_mask is not None else None
                    with torch.no_grad():
                        sub_out = self.base_model(input_ids=sub_ids, attention_mask=sub_mask, use_cache=True)
                    for l in range(self.b, self.t + 1):
                        k_s, v_s = extract_kv_for_layer(sub_out.past_key_values, l)
                        all_k_states[l].append(k_s)
                        all_v_states[l].append(v_s)
                        
                passage_cache_map = {l: (torch.cat(all_k_states[l], dim=0), torch.cat(all_v_states[l], dim=0)) for l in range(self.b, self.t + 1)}
            else:
                with torch.no_grad():
                    outputs = self.base_model(input_ids=passage_ids, attention_mask=attention_mask, use_cache=True)
                passage_cache_map = {l: extract_kv_for_layer(outputs.past_key_values, l) for l in range(self.b, self.t + 1)}
        finally:
            self.base_model.train(was_training)
        
        custom_cache = DynamicCache()
        
        # We populate custom_cache only for layers b..t
        num_layers = self.base_model.config.n_layer if self.model_type == "gpt2" else self.base_model.config.num_hidden_layers
        
        for l in range(num_layers):
            if self.b <= l <= self.t:
                k_state, v_state = passage_cache_map[l]
                
                num_heads, _, head_dim = k_state.shape[1], k_state.shape[2], k_state.shape[3]
                
                # Reshape to separate batch_size and k: [batch_size, k, num_heads, passage_len, head_dim]
                k_state = k_state.view(batch_size, k, num_heads, passage_len, head_dim)
                v_state = v_state.view(batch_size, k, num_heads, passage_len, head_dim)
                
                # Permute to make num_heads second dimension: [batch_size, num_heads, k, passage_len, head_dim]
                k_state = k_state.permute(0, 2, 1, 3, 4)
                v_state = v_state.permute(0, 2, 1, 3, 4)
                
                # Concatenate all k passages along sequence dimension: [batch_size, num_heads, k * passage_len, head_dim]
                k_concat = k_state.reshape(batch_size, num_heads, k * passage_len, head_dim)
                v_concat = v_state.reshape(batch_size, num_heads, k * passage_len, head_dim)
                
                # Write to the custom cache safely
                safe_update_dynamic_cache(custom_cache, k_concat, v_concat, l)
                
        return custom_cache

    def forward(self, query_ids, custom_past_key_values, labels=None, attention_mask=None):
        """
        Runs query tokens through the model, with pre-encoded passage KV states cross-attended in layers b..t.
        query_ids: [batch_size, query_len]
        custom_past_key_values: pre-encoded passage cache
        """
        batch_size, query_len = query_ids.shape
        if hasattr(custom_past_key_values, "key_cache") and len(custom_past_key_values.key_cache) > self.b and custom_past_key_values.key_cache[self.b] is not None and custom_past_key_values.key_cache[self.b].ndim >= 3:
            k_max_len = custom_past_key_values.key_cache[self.b].shape[2]
        else:
            k_max_len = self.k_passages * self.max_passage_len
        
        # Construct shifted position IDs
        position_ids = torch.arange(k_max_len, k_max_len + query_len, device=query_ids.device)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
        
        # Perform the forward pass with attention_mask=None so Llama dynamically computes per-layer causal mask matching prepended past_key_values length
        return self.base_model(
            input_ids=query_ids,
            past_key_values=custom_past_key_values,
            position_ids=position_ids,
            labels=labels,
            attention_mask=None
        )

    def generate(self, query_ids, custom_past_key_values, max_new_tokens=20, temperature=1.0):
        """
        Autoregressive generation loop with passage KV cache prepended and shifted position IDs.
        """
        self.eval()
        with torch.no_grad():
            batch_size, query_len = query_ids.shape
            k_max_len = self.k_passages * self.max_passage_len
            
            # Initial forward pass of the query
            position_ids = torch.arange(k_max_len, k_max_len + query_len, device=query_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
            
            outputs = self.base_model(
                input_ids=query_ids,
                past_key_values=custom_past_key_values,
                position_ids=position_ids
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            generated = [next_tokens]
            past_key_values = outputs.past_key_values
            
            # Generation loop
            for i in range(max_new_tokens - 1):
                next_pos = k_max_len + query_len + i + 1
                pos_tensor = torch.tensor([[next_pos]], device=query_ids.device).repeat(batch_size, 1)
                
                outputs = self.base_model(
                    input_ids=next_tokens,
                    past_key_values=past_key_values,
                    position_ids=pos_tensor
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                if temperature > 0:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                generated.append(next_tokens)
                past_key_values = outputs.past_key_values
                
            return torch.cat(generated, dim=-1)
