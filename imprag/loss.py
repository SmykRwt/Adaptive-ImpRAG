import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_generation_loss(logits, labels):
    """
    Computes standard causal language modeling loss, shifting logits and labels.
    Only computes loss on tokens where label != -100.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

class MultiLabelNCELoss(nn.Module):
    """
    Implements the Multi-Label Noise Contrastive Estimation (NCE) loss (Eq 3 in the paper)
    for the warmup training stage:
    J_ret(q, C) = - sum_{p in P(q)} log( exp(s(q,p) / tau) / sum_{p' in C} exp(s(q,p') / tau) )
    """
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, positive_mask):
        """
        scores: Similarity scores of shape [batch_size, num_candidates]
        positive_mask: Boolean or Float tensor of shape [batch_size, num_candidates]
                       where positive_mask[i, j] is True (or 1) if candidate j is positive for query i.
        """
        scaled_scores = scores / self.temperature
        
        # Log-denominator: logsumexp over all candidates for each query [batch_size, 1]
        log_denom = torch.logsumexp(scaled_scores, dim=-1, keepdim=True)
        
        # Log probabilities for each candidate: [batch_size, num_candidates]
        log_probs = scaled_scores - log_denom
        
        pos_mask = positive_mask.float()
        # Sum log probs of all positive passages for each query
        pos_log_probs = (log_probs * pos_mask).sum(dim=-1)
        num_positives = pos_mask.sum(dim=-1).clamp(min=1.0)
        
        # Average across positives per query, then average over the batch
        loss_per_query = - (pos_log_probs / num_positives)
        return loss_per_query.mean()

class SelfDistillationLoss(nn.Module):
    """
    Implements language model perplexity distillation (Eq 4, 5, 6 in the paper)
    for the self-distillation stage:
    Computes the KL-divergence between:
    - Teacher distribution PT(p | q, r) = Softmax(log P_LM(r | p, q) / tau_t) (detached)
    - Predicted retriever distribution PR(p | q) = Softmax(s(q, p) / tau_r)
    J_ret(q, C) = KL( PT(p | q, r) || PR(p | q) )
    """
    def __init__(self, tau_t=1.0, tau_r=1.0):
        super().__init__()
        self.tau_t = tau_t
        self.tau_r = tau_r

    def forward(self, scores, log_probs_lm, candidate_mask=None):
        """
        scores: Retriever similarity scores of shape [batch_size, num_candidates]
        log_probs_lm: Response log-likelihood under the LM for each candidate passage.
                      Shape [batch_size, num_candidates]
        candidate_mask: Optional boolean tensor [batch_size, num_candidates]
        """
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask, -1e9)
            log_probs_lm = log_probs_lm.masked_fill(~candidate_mask, -1e9)
            
        # Target soft distribution: softmax over scaled LM log likelihoods (detached from grad)
        P_T = F.softmax(log_probs_lm / self.tau_t, dim=-1).detach()
        
        # Log of predicted retriever distribution
        log_P_R = F.log_softmax(scores / self.tau_r, dim=-1)
        
        # Log of target distribution (add epsilon to prevent log(0))
        log_P_T = torch.log(P_T + 1e-12)
        
        # KL Divergence: sum_{p} P_T(p) * (log P_T(p) - log P_R(p))
        kl_div = torch.sum(P_T * (log_P_T - log_P_R), dim=-1)
        
        # Average KL-divergence over the batch
        return kl_div.mean()
