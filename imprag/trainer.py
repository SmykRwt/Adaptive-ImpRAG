import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.cache_utils import DynamicCache
from imprag.loss import MultiLabelNCELoss, SelfDistillationLoss, compute_generation_loss
from imprag.model import extract_kv_for_layer, safe_update_dynamic_cache, pad_cache_to_num_layers

class ImpRAGTrainer:
    """
    Two-stage trainer for ImpRAG matching Section 3.2 in the paper:
    1. Warmup Stage (Epochs 1..warmup_epochs):
       - Multi-Label NCE loss (Eq 3) over pseudo-positives P(q) and hard + in-batch negatives.
    2. Self-Distillation Stage (Remaining epochs):
       - Self-distillation KL-divergence loss (Eq 4-6) against LM response log-likelihoods.
    3. Joint Multi-Task Objective (Eq 2):
       - J = J_gen(r | q, C) + lambda_ret * J_ret(q, C).
    """
    def __init__(self, model, optimizer, warmup_epochs=3, total_epochs=5, lambda_ret=1.0, 
                 device="cpu", tau_t=1.0, tau_r=1.0, accumulation_steps=1, use_amp=False, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lambda_ret = lambda_ret
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.scheduler = scheduler
        
        self.warmup_loss_fn = MultiLabelNCELoss(temperature=0.05)
        self.distill_loss_fn = SelfDistillationLoss(tau_t=tau_t, tau_r=tau_r)
        
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        else:
            self.scaler = None

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        total_gen_loss = 0.0
        total_ret_loss = 0.0
        
        is_warmup = epoch < self.warmup_epochs
        stage_name = "WARMUP (NCE)" if is_warmup else "SELF-DISTILLATION (KL)"
        
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1
        
        self.optimizer.zero_grad()
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        
        for batch_idx, batch in enumerate(dataloader):
            query_ids = batch["query_ids"].to(self.device)
            full_ids = batch["full_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            candidate_passage_ids = batch["candidate_passage_ids"].to(self.device)
            positive_mask = batch["positive_mask"].to(self.device)
            primary_pos_ids = batch["primary_positive_ids"].to(self.device)
            
            batch_size = query_ids.shape[0]
            if batch_size < 1:
                continue
                
            query_mask = batch["query_attention_mask"].to(self.device)
            full_mask = batch["full_attention_mask"].to(self.device)
            cand_mask = batch["candidate_passage_attention_mask"].to(self.device)
            primary_pos_mask = batch["primary_positive_attention_mask"].to(self.device)
            
            autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.use_amp else torch.cuda.amp.autocast(enabled=False)
            
            with autocast_context:
                # 1. Compute Retriever Embeddings
                E_q = model_obj.get_retriever_embeddings(query_ids, attention_mask=query_mask, is_query=True)
                E_p = model_obj.get_retriever_embeddings(candidate_passage_ids, attention_mask=cand_mask, is_query=False)
                
                # Normalize embeddings for cosine/dot product
                E_q_norm = F.normalize(E_q, p=2, dim=-1)
                E_p_norm = F.normalize(E_p, p=2, dim=-1)
                
                if is_distributed:
                    gathered_Ep = [torch.zeros_like(E_p_norm) for _ in range(world_size)]
                    dist.all_gather(gathered_Ep, E_p_norm)
                    gathered_Ep_norm = torch.cat(gathered_Ep, dim=0)
                    scores = torch.matmul(E_q_norm, gathered_Ep_norm.transpose(0, 1))
                    
                    # Pad positive_mask across gathered items
                    gathered_mask = [torch.zeros_like(positive_mask) for _ in range(world_size)]
                    gathered_mask[rank] = positive_mask
                    full_pos_mask = torch.cat(gathered_mask, dim=1)
                else:
                    scores = torch.matmul(E_q_norm, E_p_norm.transpose(0, 1))  # [B, Num_candidates]
                    full_pos_mask = positive_mask
                
                # 2. Retrieval Loss (Section 3.2.1)
                if is_warmup:
                    # Multi-Label NCE Loss (Eq 3)
                    ret_loss = self.warmup_loss_fn(scores, full_pos_mask)
                else:
                    # Self-Distillation Loss (Eq 4-6)
                    # Evaluate response log-likelihoods conditioned on each candidate passage
                    num_cands = candidate_passage_ids.shape[0]
                    eval_cand_ids = candidate_passage_ids[:min(num_cands, batch_size * 5)]
                    scores_subset = scores[:, :eval_cand_ids.shape[0]]
                    
                    was_tr = model_obj.base_model.training
                    model_obj.base_model.eval()
                    try:
                        with torch.no_grad():
                            cand_cache = model_obj.base_model(
                                input_ids=eval_cand_ids,
                                use_cache=True
                            ).past_key_values
                            
                            num_layers = model_obj._get_num_layers()
                            log_probs_list = []
                            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                            
                            for c_idx in range(eval_cand_ids.shape[0]):
                                single_cache = DynamicCache()
                                sample_k = None
                                for l in range(num_layers):
                                    if model_obj.b <= l <= model_obj.t:
                                        k_l, v_l = extract_kv_for_layer(cand_cache, l)
                                        rep_k = k_l[c_idx:c_idx+1].repeat(batch_size, 1, 1, 1)
                                        rep_v = v_l[c_idx:c_idx+1].repeat(batch_size, 1, 1, 1)
                                        safe_update_dynamic_cache(single_cache, rep_k, rep_v, l)
                                        sample_k = rep_k
                                        
                                if sample_k is not None:
                                    pad_cache_to_num_layers(
                                        single_cache,
                                        num_layers=num_layers,
                                        batch_size=batch_size,
                                        num_heads=sample_k.shape[1],
                                        head_dim=sample_k.shape[3],
                                        device=sample_k.device,
                                        dtype=sample_k.dtype
                                    )
                                    
                                out_lm = model_obj(query_ids=full_ids, custom_past_key_values=single_cache)
                                shift_logits = out_lm.logits[..., :-1, :].contiguous()
                                shift_labels = labels[..., 1:].contiguous()
                                
                                token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                                token_loss = token_loss.view(batch_size, -1)
                                mask = (shift_labels != -100).float()
                                seq_loss = (token_loss * mask).sum(dim=-1)
                                log_probs_list.append(-seq_loss)
                                
                            log_probs_lm = torch.stack(log_probs_list, dim=1)  # [batch_size, num_eval_cands]
                    finally:
                        model_obj.base_model.train(was_tr)
                        
                    ret_loss = self.distill_loss_fn(scores_subset, log_probs_lm)
                    
                # 3. Joint Generator Forward Pass (Section 3.2 Eq 2)
                top_pos_cache = model_obj.encode_passages(
                    primary_pos_ids, 
                    attention_mask=primary_pos_mask,
                    k_passages=1,
                    encoding_mode="concatenated"
                )
                
                gen_outputs = model_obj(
                    query_ids=full_ids,
                    custom_past_key_values=top_pos_cache,
                    labels=labels,
                    attention_mask=full_mask
                )
                
                gen_loss = gen_outputs.loss
                loss = gen_loss + self.lambda_ret * ret_loss
                
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps
                    
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
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
                      f"Total = {loss.item() * self.accumulation_steps:.4f} (Gen = {gen_loss.item():.4f}, Ret = {ret_loss.item():.4f})")
                
        avg_loss = total_loss / max(len(dataloader), 1)
        avg_gen = total_gen_loss / max(len(dataloader), 1)
        avg_ret = total_ret_loss / max(len(dataloader), 1)
        return avg_loss, avg_gen, avg_ret

    def evaluate(self, dataloader, tokenizer, k_top=5):
        """
        Evaluates Exact Match (EM) and Recall@1 / Recall@k matching the evaluation setup in Section 4.1.
        """
        self.model.eval()
        exact_matches = 0
        retrieval_recalls = 0
        total_samples = 0
        
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        
        with torch.no_grad():
            for batch in dataloader:
                query_ids = batch["query_ids"].to(self.device)
                cand_ids = batch["candidate_passage_ids"].to(self.device)
                positive_mask = batch["positive_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                batch_size = query_ids.shape[0]
                query_mask = batch["query_attention_mask"].to(self.device)
                cand_mask = batch["candidate_passage_attention_mask"].to(self.device)
                
                E_q = model_obj.get_retriever_embeddings(query_ids, attention_mask=query_mask, is_query=True)
                E_p = model_obj.get_retriever_embeddings(cand_ids, attention_mask=cand_mask, is_query=False)
                
                E_q_norm = F.normalize(E_q, p=2, dim=-1)
                E_p_norm = F.normalize(E_p, p=2, dim=-1)
                
                scores = torch.matmul(E_q_norm, E_p_norm.transpose(0, 1))  # [batch_size, num_cands]
                
                _, topk_indices = torch.topk(scores, k=min(k_top, scores.shape[1]), dim=-1)
                
                # Check Recall@k
                for i in range(batch_size):
                    top_candidates = topk_indices[i].tolist()
                    is_recalled = any(positive_mask[i, c_idx].item() for c_idx in top_candidates if c_idx < positive_mask.shape[1])
                    if is_recalled:
                        retrieval_recalls += 1
                        
                # Perform Generation with Top-k Retrieved Passages
                for i in range(batch_size):
                    top_k_indices_i = topk_indices[i][:k_top]
                    selected_cand_ids = cand_ids[top_k_indices_i].unsqueeze(0)  # [1, k, seq_len]
                    
                    custom_cache = model_obj.encode_passages(
                        selected_cand_ids, 
                        k_passages=len(top_k_indices_i),
                        encoding_mode="concatenated"
                    )
                    
                    gen_tokens = model_obj.generate(
                        query_ids=query_ids[i:i+1],
                        custom_past_key_values=custom_cache,
                        max_new_tokens=15
                    )
                    
                    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip().lower()
                    
                    # Extract ground-truth answer
                    ans_tokens = labels[i][labels[i] != -100]
                    ans_text = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip().lower()
                    
                    if ans_text in gen_text or gen_text == ans_text:
                        exact_matches += 1
                        
                    total_samples += 1
                    
        em_score = exact_matches / max(total_samples, 1)
        recall_score = retrieval_recalls / max(total_samples, 1)
        return em_score, recall_score
