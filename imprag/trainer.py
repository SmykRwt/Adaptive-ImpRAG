import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.cache_utils import DynamicCache
from imprag.loss import SelfDistillationLoss

from imprag.model import extract_kv_for_layer, safe_update_dynamic_cache

class ImpRAGTrainer:
    """
    Two-stage trainer for ImpRAG using in-batch negatives with DDP multi-GPU support:
    - Epochs 1..warmup_epochs: Warmup stage using standard cross-entropy loss over gathered in-batch dot products.
    - Remaining epochs: Self-distillation stage using language model perplexity distillation over local in-batch candidates.
    """
    def __init__(self, model, optimizer, warmup_epochs=3, total_epochs=5, lambda_ret=1.0, device="cpu", tau_t=1.0, tau_r=1.0, accumulation_steps=1, use_amp=False, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lambda_ret = lambda_ret
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.scheduler = scheduler
        
        self.warmup_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = SelfDistillationLoss(tau_t=tau_t, tau_r=tau_r)
        
        if self.use_amp:
            # GradScaler is not required or supported for bfloat16 in PyTorch (bfloat16 has 8-bit exponent)
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        else:
            self.scaler = None

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        total_gen_loss = 0.0
        total_ret_loss = 0.0
        
        is_warmup = epoch < self.warmup_epochs
        stage_name = "WARMUP" if is_warmup else "SELF-DISTILLATION"
        
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1
        
        self.optimizer.zero_grad()
        
        # Unpack DDP wrapper if present
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        
        for batch_idx, batch in enumerate(dataloader):
            # Move inputs to device
            query_ids = batch["query_ids"].to(self.device)
            full_ids = batch["full_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            candidate_passage_ids = batch["candidate_passage_ids"].to(self.device)
            
            batch_size = query_ids.shape[0]
            if batch_size < 2:
                continue
                
            query_attention_mask = batch["query_attention_mask"].to(self.device)
            candidate_passage_attention_mask = batch["candidate_passage_attention_mask"].to(self.device)
            
            autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.use_amp else torch.cuda.amp.autocast(enabled=False)
            
            with autocast_context:
                E_q = model_obj.get_retriever_embeddings(query_ids, attention_mask=query_attention_mask, is_query=True)
                E_p = model_obj.get_retriever_embeddings(candidate_passage_ids, attention_mask=candidate_passage_attention_mask, is_query=False)
                
                E_q_norm = F.normalize(E_q, p=2, dim=-1)
                E_p_norm = F.normalize(E_p, p=2, dim=-1)
                
                if is_distributed:
                    # All-gather positive passage embeddings across all GPUs for true paper-scale negatives
                    gathered_Ep = [torch.zeros_like(E_p_norm) for _ in range(world_size)]
                    dist.all_gather(gathered_Ep, E_p_norm)
                    gathered_Ep_norm = torch.cat(gathered_Ep, dim=0)  # [N*B, D]
                    
                    scores = torch.matmul(E_q_norm, gathered_Ep_norm.transpose(0, 1)) / 0.05  # [B, N*B]
                else:
                    scores = torch.matmul(E_q_norm, E_p_norm.transpose(0, 1)) / 0.05  # [B, B]
                
                if is_warmup:
                    if is_distributed:
                        # Diagonal index targets corresponding to the local rank offset
                        targets = torch.arange(rank * batch_size, (rank + 1) * batch_size, device=self.device)
                    else:
                        targets = torch.arange(batch_size, device=self.device)
                    ret_loss = self.warmup_loss_fn(scores, targets)
                else:
                    # Compute self-distillation locally to avoid VRAM exhaustion
                    scores_local = torch.matmul(E_q_norm, E_p_norm.transpose(0, 1)) / 0.05  # [B, B]
                    
                    replicated_full_ids = full_ids.repeat(batch_size, 1)
                    replicated_labels = labels.repeat(batch_size, 1)
                    
                    was_tr = model_obj.base_model.training
                    model_obj.base_model.eval()
                    try:
                        with torch.no_grad():
                            passage_outputs = model_obj.base_model(
                                input_ids=candidate_passage_ids,
                                use_cache=True
                            )
                            passage_cache = passage_outputs.past_key_values
                    finally:
                        model_obj.base_model.train(was_tr)
                        
                        replicated_cache = DynamicCache()
                        num_layers = model_obj.base_model.config.n_layer if model_obj.model_type == "gpt2" else model_obj.base_model.config.num_hidden_layers
                        
                        for l in range(num_layers):
                            if model_obj.b <= l <= model_obj.t:
                                k, v = extract_kv_for_layer(passage_cache, l)
                                rep_k = k.repeat_interleave(batch_size, dim=0)
                                rep_v = v.repeat_interleave(batch_size, dim=0)
                                safe_update_dynamic_cache(replicated_cache, rep_k, rep_v, l)
                                
                        k_max_len = model_obj.k_passages * model_obj.max_passage_len
                        query_len = replicated_full_ids.shape[1]
                        position_ids = torch.arange(k_max_len, k_max_len + query_len, device=self.device)
                        position_ids = position_ids.unsqueeze(0).repeat(batch_size * batch_size, 1)
                        
                        outputs_rep = model_obj.base_model(
                            input_ids=replicated_full_ids,
                            past_key_values=replicated_cache,
                            position_ids=position_ids
                        )
                        
                        logits_rep = outputs_rep.logits
                        shift_logits = logits_rep[..., :-1, :].contiguous()
                        shift_labels = replicated_labels[..., 1:].contiguous()
                        
                        loss_fct = nn.CrossEntropyLoss(reduction="none")
                        loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        loss_per_token = loss_per_token.view(batch_size * batch_size, -1)
                        
                        mask = (shift_labels != -100).float()
                        loss_per_seq = (loss_per_token * mask).sum(dim=-1)
                        
                        log_probs_lm = -loss_per_seq.view(batch_size, batch_size).transpose(0, 1)
                    
                    ret_loss = self.distill_loss_fn(scores_local, log_probs_lm)
                    
                # 3. Main Generator Forward Pass (Query token loss)
                topk_passage_ids = candidate_passage_ids[:batch_size * model_obj.k_passages]
                topk_cache = model_obj.encode_passages(topk_passage_ids)
                
                outputs = model_obj(
                    query_ids=query_ids,
                    custom_past_key_values=topk_cache,
                    labels=labels,
                    attention_mask=query_attention_mask
                )
                
                gen_loss = outputs.loss
                loss = gen_loss + self.lambda_ret * ret_loss
                
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            if self.use_amp and self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if self.use_amp and self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                    
            total_loss += loss.item() * self.accumulation_steps
            total_gen_loss += gen_loss.item()
            total_ret_loss += ret_loss.item()
            
            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch+1} [{stage_name}] Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss = {loss.item() * self.accumulation_steps:.4f} (Gen = {gen_loss.item():.4f}, Ret = {ret_loss.item():.4f})")
                
        avg_loss = total_loss / max(len(dataloader), 1)
        avg_gen_loss = total_gen_loss / max(len(dataloader), 1)
        avg_ret_loss = total_ret_loss / max(len(dataloader), 1)
        return avg_loss, avg_gen_loss, avg_ret_loss

    def evaluate(self, dataloader, tokenizer, k_top=5):
        """
        Evaluate the generation exact match (EM) and retrieval recall.
        """
        self.model.eval()
        exact_matches = 0
        total_samples = 0
        retrieval_recalls = 0
        
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        
        with torch.no_grad():
            for batch in dataloader:
                query_ids = batch["query_ids"].to(self.device)
                candidate_passage_ids = batch["candidate_passage_ids"].to(self.device)
                positive_mask = batch["positive_mask"].to(self.device)
                
                batch_size = query_ids.shape[0]
                num_candidates = candidate_passage_ids.shape[0] // batch_size
                
                query_attention_mask = batch["query_attention_mask"].to(self.device)
                candidate_passage_attention_mask = batch["candidate_passage_attention_mask"].to(self.device)
                
                E_q = model_obj.get_retriever_embeddings(query_ids, attention_mask=query_attention_mask, is_query=True)
                E_p = model_obj.get_retriever_embeddings(candidate_passage_ids, attention_mask=candidate_passage_attention_mask, is_query=False)
                
                E_q_norm = F.normalize(E_q, p=2, dim=-1)
                E_p_norm = F.normalize(E_p, p=2, dim=-1)
                
                dim = E_q.shape[-1]
                E_p_reshaped = E_p_norm.view(batch_size, num_candidates, dim)
                scores = torch.bmm(E_q_norm.unsqueeze(1), E_p_reshaped.transpose(1, 2)).squeeze(1)
                
                _, topk_indices = torch.topk(scores, k=1, dim=-1)
                
                for i in range(batch_size):
                    ret_idx = topk_indices[i, 0].item()
                    if positive_mask.shape[1] == batch_size:
                        if ret_idx == i:
                            retrieval_recalls += 1
                    else:
                        if positive_mask[i, ret_idx].item() is True:
                            retrieval_recalls += 1
                            
                k_passages = min(model_obj.k_passages, num_candidates)
                _, topk_indices_gen = torch.topk(scores, k=k_passages, dim=-1)
                
                selected_passages_list = []
                for i in range(batch_size):
                    item_candidates = candidate_passage_ids[i * num_candidates : (i + 1) * num_candidates]
                    selected_passages = item_candidates[topk_indices_gen[i]]
                    selected_passages_list.append(selected_passages)
                    
                selected_passage_ids = torch.cat(selected_passages_list, dim=0)
                
                orig_k = model_obj.k_passages
                model_obj.k_passages = k_passages
                try:
                    topk_cache = model_obj.encode_passages(selected_passage_ids)
                    generated_tokens = model_obj.generate(
                        query_ids=query_ids,
                        custom_past_key_values=topk_cache,
                        max_new_tokens=15
                    )
                finally:
                    model_obj.k_passages = orig_k
                
                gen_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                labels = batch["labels"].to(self.device)
                
                for i in range(batch_size):
                    ans_tokens = labels[i][labels[i] != -100]
                    ans_text = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()
                    gen_text = gen_texts[i].strip()
                    
                    if gen_text.lower() == ans_text.lower() or ans_text.lower() in gen_text.lower():
                        exact_matches += 1
                        
                    total_samples += 1
                    
        em_score = exact_matches / max(total_samples, 1)
        recall_score = retrieval_recalls / max(total_samples, 1)
        return em_score, recall_score
