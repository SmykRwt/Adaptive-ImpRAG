import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Native PyTorch Low-Rank Adaptation (LoRA) Layer (Hu et al., 2021).
    W_effective = W_base + (alpha / r) * (B @ A)
    """
    def __init__(self, original_module, r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.base_module = original_module
        
        if hasattr(original_module, "in_features") and hasattr(original_module, "out_features"):
            self.in_features = original_module.in_features
            self.out_features = original_module.out_features
        elif hasattr(original_module, "weight"):
            self.in_features = original_module.weight.shape[0]
            self.out_features = original_module.weight.shape[1]
        else:
            self.in_features = 128
            self.out_features = 128
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Frozen base weight
        for param in self.base_module.parameters():
            param.requires_grad = False
            
        device = original_module.weight.device if hasattr(original_module, "weight") else None
        dtype = original_module.weight.dtype if hasattr(original_module, "weight") else torch.float32
        
        # Trainable low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base_module(x)
        A = self.lora_A.to(x.device, x.dtype)
        B = self.lora_B.to(x.device, x.dtype)
        lora_out = (self.dropout(x) @ A.transpose(0, 1)) @ B.transpose(0, 1)
        return base_out + self.scaling * lora_out

    def merge_weights(self):
        """Merges LoRA weights directly into base tensor for 0-latency inference."""
        weight_update = (self.lora_B @ self.lora_A) * self.scaling
        if hasattr(self.base_module, "weight"):
            if self.base_module.weight.shape == weight_update.shape:
                self.base_module.weight.data += weight_update.to(self.base_module.weight.dtype)
            elif self.base_module.weight.shape == weight_update.T.shape:
                self.base_module.weight.data += weight_update.T.to(self.base_module.weight.dtype)

def apply_lora_to_imprag(model, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "c_attn"], max_layer=7):
    """
    Applies LoRA adapters to key/query attention projections in bottom layers (0..max_layer)
    and freezes all other base parameters, satisfying Section 6.5 of the Capstone Report.
    """
    model_obj = model.module if hasattr(model, "module") else model
    base = model_obj.base_model if hasattr(model_obj, "base_model") else model_obj
    
    # Freeze all parameters
    for param in base.parameters():
        param.requires_grad = False
        
    lora_layers = {}
    
    # Identify layers list
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        layers = base.model.layers
    elif hasattr(base, "layers"):
        layers = base.layers
    elif hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        layers = base.transformer.h
    elif hasattr(base, "h"):
        layers = base.h
    else:
        layers = []
        
    for l_idx in range(min(max_layer + 1, len(layers))):
        layer = layers[l_idx]
        attn = layer.self_attn if hasattr(layer, "self_attn") else (layer.attn if hasattr(layer, "attn") else None)
        if attn is None:
            continue
            
        for module_name in target_modules:
            if hasattr(attn, module_name):
                orig_mod = getattr(attn, module_name)
                if not isinstance(orig_mod, LoRALinear):
                    lora_mod = LoRALinear(orig_mod, r=r, lora_alpha=lora_alpha)
                    setattr(attn, module_name, lora_mod)
                    lora_layers[f"layer_{l_idx}.{module_name}"] = lora_mod
                    
    total_params = sum(p.numel() for p in base.parameters())
    trainable_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
    reduction = 100.0 * (1.0 - trainable_params / max(1, total_params))
    
    print(f"Applied LoRA (r={r}, alpha={lora_alpha}) to {len(lora_layers)} modules across layers 0..{max_layer}.")
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({reduction:.2f}% parameter reduction)")
    
    return lora_layers
