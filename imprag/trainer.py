import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from imprag.loss import compute_generation_loss, MultiLabelNCELoss, SelfDistillationLoss

class ImpRAGTrainer:
    """
    Two-stage trainer for ImpRAG:
    - Epochs 1..warmup_epochs: Warmup stage using Multi-Label NCE loss with pseudo-labels.
    - Remaining epochs: Self-distillation stage using language model perplexity distillation.
    """
    def __init__(self, model, optimizer, warmup_epochs=3, total_epochs=5, lambda_ret=1.0, device="cpu", tau_t=1.0, tau_r=1.0):
        self.model = model
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lambda_ret = lambda_ret
        self.device = device
        
        self.warmup_loss_fn = MultiLabelNCELoss()
        self.distill_loss_fn = SelfDistillationLoss(tau_t=tau_t, tau_r=tau_r)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        total_gen_loss = 0.0
        total_ret_loss = 0.0
        
        is_warmup = epoch < self.warmup_epochs
        stage_name = "WARMUP" if is_warmup else "SELF-DISTILLATION"
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Move inputs to device
            query_ids = batch["query_ids"].to(self.device)
            full_ids = batch["full_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            candidate_passage_ids = batch["candidate_passage_ids"].to(self.device)
            positive_mask = batch["positive_mask"].to(self.device)
            
            batch_size, num_candidates = positive_mask.shape
            
            # --- 1. Compute Retriever Embeddings & Similarity Scores ---
            # Extract query embeddings [batch_size, dim]
            query_attention_mask = batch["query_attention_mask"].to(self.device)
            candidate_passage_attention_mask = batch["candidate_passage_attention_mask"].to(self.device)
            
            E_q = self.model.get_retriever_embeddings(query_ids, attention_mask=query_attention_mask, is_query=True)
            # Extract candidate passage embeddings [batch_size * num_candidates, dim]
            E_p = self.model.get_retriever_embeddings(candidate_passage_ids, attention_mask=candidate_passage_attention_mask, is_query=False)
            
            # L2-normalize embeddings to compute cosine similarity (InfoNCE training stability)
            E_q_norm = F.normalize(E_q, p=2, dim=-1)
            E_p_norm = F.normalize(E_p, p=2, dim=-1)
            
            # Reshape E_p to [batch_size, num_candidates, dim]
            dim = E_q.shape[-1]
            E_p_reshaped = E_p_norm.view(batch_size, num_candidates, dim)
            
            # Similarity scores: scaled cosine similarity (temperature = 0.05)
            scores = torch.bmm(E_q_norm.unsqueeze(1), E_p_reshaped.transpose(1, 2)).squeeze(1) / 0.05
            
            # --- 2. Compute Retrieval Loss (NCE or Distillation) ---
            if is_warmup:
                # Warmup stage: Multi-Label NCE loss
                ret_loss = self.warmup_loss_fn(scores, positive_mask)
            else:
                # Self-distillation stage: Perplexity distillation
                # To do this in a single forward pass, we replicate the query/response
                # and compute the log-likelihood of each candidate passage.
                
                # Replicate full_ids and labels for each candidate: [batch_size * num_candidates, seq_len]
                replicated_full_ids = full_ids.repeat_interleave(num_candidates, dim=0)
                replicated_labels = labels.repeat_interleave(num_candidates, dim=0)
                
                with torch.no_grad():
                    # Get KV states for all candidates: [batch_size * num_candidates, num_heads, passage_len, head_dim]
                    # We can run the base model up to layer t to get their KV states.
                    passage_outputs = self.model.base_model(
                        input_ids=candidate_passage_ids,
                        use_cache=True
                    )
                    passage_cache = passage_outputs.past_key_values
                    
                    # Construct a cache for the replicated queries
                    replicated_cache = DynamicCache()
                    num_layers = self.model.base_model.config.n_layer if self.model.model_type == "gpt2" else self.model.base_model.config.num_hidden_layers
                    
                    for l in range(num_layers):
                        if self.model.b <= l <= self.model.t:
                            layer_cache = passage_cache.layers[l]
                            replicated_cache.update(layer_cache.keys, layer_cache.values, l)
                            
                    # Run query forward pass with replicated cache
                    # Shifted positions for query
                    k_max_len = self.model.k_passages * self.model.max_passage_len
                    query_len = replicated_full_ids.shape[1]
                    position_ids = torch.arange(k_max_len, k_max_len + query_len, device=self.device)
                    position_ids = position_ids.unsqueeze(0).repeat(batch_size * num_candidates, 1)
                    
                    # Forward pass on replicated queries to compute response likelihood
                    outputs_rep = self.model.base_model(
                        input_ids=replicated_full_ids,
                        past_key_values=replicated_cache,
                        position_ids=position_ids
                    )
                    
                    # Calculate negative log likelihood per sequence
                    logits_rep = outputs_rep.logits
                    shift_logits = logits_rep[..., :-1, :].contiguous()
                    shift_labels = replicated_labels[..., 1:].contiguous()
                    
                    loss_fct = nn.CrossEntropyLoss(reduction="none")
                    loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss_per_token = loss_per_token.view(batch_size * num_candidates, -1)
                    
                    # Sum over response tokens
                    mask = (shift_labels != -100).float()
                    loss_per_seq = (loss_per_token * mask).sum(dim=-1)
                    
                    log_probs_lm = -loss_per_seq.view(batch_size, num_candidates)
                
                # Distillation KL loss
                ret_loss = self.distill_loss_fn(scores, log_probs_lm)
                
            # --- 3. Compute Reader Generation Loss ---
            # Retrieve the top-k passages based on current retriever scores
            _, topk_indices = torch.topk(scores, k=self.model.k_passages, dim=-1)
            
            selected_passages_list = []
            for i in range(batch_size):
                # Candidate passages for query i
                item_candidates = candidate_passage_ids[i * num_candidates : (i + 1) * num_candidates]
                selected_passages = item_candidates[topk_indices[i]]  # [k_passages, passage_len]
                selected_passages_list.append(selected_passages)
                
            # Concatenate selected passages: [batch_size * k, passage_len]
            selected_passage_ids = torch.cat(selected_passages_list, dim=0)
            
            # Encode these selected passages
            topk_cache = self.model.encode_passages(selected_passage_ids)
            
            # Run forward pass of query with top-k cache
            outputs = self.model(
                query_ids=full_ids,
                custom_past_key_values=topk_cache,
                labels=labels
            )
            
            # Generation loss
            gen_loss = outputs.loss
            
            # --- 4. Optimization ---
            loss = gen_loss + self.lambda_ret * ret_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                # Clear gradients and skip to avoid corrupting model weights
                self.optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_gen_loss += gen_loss.item()
                total_ret_loss += ret_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} [{stage_name}] Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss = {loss.item():.4f} (Gen = {gen_loss.item():.4f}, Ret = {ret_loss.item():.4f})")
                
        avg_loss = total_loss / len(dataloader)
        avg_gen_loss = total_gen_loss / len(dataloader)
        avg_ret_loss = total_ret_loss / len(dataloader)
        return avg_loss, avg_gen_loss, avg_ret_loss

    def evaluate(self, dataloader, tokenizer, k_top=5):
        """
        Evaluate the generation exact match (EM) and retrieval recall.
        """
        self.model.eval()
        exact_matches = 0
        total_samples = 0
        retrieval_recalls = 0
        
        with torch.no_grad():
            for batch in dataloader:
                query_ids = batch["query_ids"].to(self.device)
                candidate_passage_ids = batch["candidate_passage_ids"].to(self.device)
                positive_mask = batch["positive_mask"].to(self.device)
                
                batch_size, num_candidates = positive_mask.shape
                
                # 1. Evaluate retrieval
                query_attention_mask = batch["query_attention_mask"].to(self.device)
                candidate_passage_attention_mask = batch["candidate_passage_attention_mask"].to(self.device)
                
                E_q = self.model.get_retriever_embeddings(query_ids, attention_mask=query_attention_mask, is_query=True)
                E_p = self.model.get_retriever_embeddings(candidate_passage_ids, attention_mask=candidate_passage_attention_mask, is_query=False)
                
                # L2-normalize embeddings for evaluation
                E_q_norm = F.normalize(E_q, p=2, dim=-1)
                E_p_norm = F.normalize(E_p, p=2, dim=-1)
                
                dim = E_q.shape[-1]
                E_p_reshaped = E_p_norm.view(batch_size, num_candidates, dim)
                scores = torch.bmm(E_q_norm.unsqueeze(1), E_p_reshaped.transpose(1, 2)).squeeze(1)
                
                # Check if the top-1 retrieved passage is positive (Warmup recall or basic recall)
                _, topk_indices = torch.topk(scores, k=1, dim=-1) # [batch_size, 1]
                
                for i in range(batch_size):
                    ret_idx = topk_indices[i, 0].item()
                    if positive_mask[i, ret_idx].item() is True:
                        retrieval_recalls += 1
                        
                # 2. Evaluate generation (Exact Match)
                # Select top-k passages for generation
                _, topk_indices_gen = torch.topk(scores, k=self.model.k_passages, dim=-1)
                
                selected_passages_list = []
                for i in range(batch_size):
                    item_candidates = candidate_passage_ids[i * num_candidates : (i + 1) * num_candidates]
                    selected_passages = item_candidates[topk_indices_gen[i]]
                    selected_passages_list.append(selected_passages)
                    
                selected_passage_ids = torch.cat(selected_passages_list, dim=0)
                topk_cache = self.model.encode_passages(selected_passage_ids)
                
                # Generate responses
                generated_tokens = self.model.generate(
                    query_ids=query_ids,
                    custom_past_key_values=topk_cache,
                    max_new_tokens=15
                )
                
                # Decode responses
                gen_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                # Ground truth answers
                labels = batch["labels"].to(self.device)
                
                for i in range(batch_size):
                    # Extract answer string by decoding response labels (non -100)
                    ans_tokens = labels[i][labels[i] != -100]
                    ans_text = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()
                    gen_text = gen_texts[i].strip()
                    
                    # Check exact match
                    if gen_text.lower() == ans_text.lower() or ans_text.lower() in gen_text.lower():
                        exact_matches += 1
                        
                    total_samples += 1
                    
        em_score = exact_matches / total_samples
        recall_score = retrieval_recalls / total_samples
        return em_score, recall_score
