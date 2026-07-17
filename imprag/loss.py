import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_generation_loss(logits, labels):
    """
    Computes standard causal language modeling loss, shifting logits and labels.
    Only computes loss on tokens where label != -100.
    """
    # Shift logits and labels for autoregressive prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

class MultiLabelNCELoss(nn.Module):
    """
    Implements the Multi-Label Noise Contrastive Estimation (NCE) loss (Eq 3 in the paper)
    for the warmup training stage:
    J_ret(q, C) = - sum_{p in P(q)} log( exp(s(q,p)) / (exp(s(q,p)) + sum_{p' in N(q)} exp(s(q,p'))) )
    """
    def __init__(self):
        super().__init__()

    def forward(self, scores, positive_mask):
        """
        scores: Similarity scores of shape [batch_size, num_candidates]
        positive_mask: Boolean tensor of shape [batch_size, num_candidates]
                       where positive_mask[i, j] is True if candidate j is positive for query i.
        """
        batch_size, num_candidates = scores.shape
        loss = 0.0
        total_positives = 0
        
        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i])[0]
            neg_indices = torch.where(~positive_mask[i])[0]
            
            if len(pos_indices) == 0:
                continue
                
            neg_scores = scores[i, neg_indices]  # [num_negatives]
            
            for pos_idx in pos_indices:
                pos_score = scores[i, pos_idx]
                # Logsumexp over (pos_score, neg_scores)
                denom_scores = torch.cat([pos_score.unsqueeze(0), neg_scores])
                log_prob = pos_score - torch.logsumexp(denom_scores, dim=0)
                loss = loss - log_prob
                total_positives += 1
                
        return loss / max(total_positives, 1)

class SelfDistillationLoss(nn.Module):
    """
    Implements language model perplexity distillation (Eq 4, 5, 6 in the paper)
    for the self-distillation stage:
    Computes the KL-divergence between:
    - Teacher distribution PT(p | q, r) (derived from LM likelihoods detached from grad)
    - Predicted retriever distribution PR(p | q) (derived from retrieval scores with grad)
    """
    def __init__(self, tau_t=1.0, tau_r=1.0):
        super().__init__()
        self.tau_t = tau_t
        self.tau_r = tau_r

    def forward(self, scores, log_probs_lm):
        """
        scores: Retriever similarity scores of shape [batch_size, num_candidates]
        log_probs_lm: Response log-likelihood under the LM for each candidate passage.
                      Shape [batch_size, num_candidates]
        """
        # Target soft distribution: softmax over scaled LM log likelihoods (detached)
        P_T = F.softmax(log_probs_lm / self.tau_t, dim=-1).detach()
        
        # Log of predicted retriever distribution
        log_P_R = F.log_softmax(scores / self.tau_r, dim=-1)
        
        # Avoid log(0) in target distribution
        log_P_T = torch.log(P_T + 1e-10)
        
        # KL Divergence: sum_{p} P_T(p) * (log P_T(p) - log P_R(p))
        kl_div = torch.sum(P_T * (log_P_T - log_P_R), dim=-1)
        
        # Average KL-divergence over the batch
        return kl_div.mean()
