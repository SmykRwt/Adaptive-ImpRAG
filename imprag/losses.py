from typing import Optional

import torch
import torch.nn.functional as F


def multi_label_nce_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    scores: [batch, candidates]
    positive_mask: [batch, candidates] with 1 for pseudo positives
    negative_mask: optional [batch, candidates] with 1 for negatives; defaults to logical not positive
    """
    if negative_mask is None:
        negative_mask = (~positive_mask.bool()).to(scores.dtype)
    else:
        negative_mask = negative_mask.to(scores.dtype)

    positive_mask = positive_mask.to(scores.dtype)
    losses = []
    for row_scores, row_pos, row_neg in zip(scores, positive_mask, negative_mask):
        pos_indices = torch.nonzero(row_pos > 0, as_tuple=False).squeeze(-1)
        neg_indices = torch.nonzero(row_neg > 0, as_tuple=False).squeeze(-1)
        if pos_indices.numel() == 0:
            continue
        neg_logits = row_scores.index_select(0, neg_indices) if neg_indices.numel() else row_scores.new_empty(0)
        neg_term = torch.logsumexp(neg_logits, dim=0) if neg_logits.numel() else torch.tensor(0.0, device=row_scores.device)
        for pos_idx in pos_indices:
            pos_logit = row_scores[pos_idx]
            denom = torch.logsumexp(torch.cat([pos_logit.view(1), neg_logits]), dim=0) if neg_logits.numel() else pos_logit
            losses.append(-(pos_logit - denom))
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def kl_retrieval_loss(
    retrieval_scores: torch.Tensor,
    lm_log_likelihoods: torch.Tensor,
    retrieval_temperature: float = 1.0,
    target_temperature: float = 1.0,
) -> torch.Tensor:
    target = torch.softmax(lm_log_likelihoods / target_temperature, dim=-1).detach()
    predicted_log_probs = torch.log_softmax(retrieval_scores / retrieval_temperature, dim=-1)
    return F.kl_div(predicted_log_probs, target, reduction="batchmean")


def joint_loss(
    generation_loss: torch.Tensor,
    retrieval_loss: Optional[torch.Tensor],
    retrieval_loss_weight: float,
) -> torch.Tensor:
    if retrieval_loss is None:
        return generation_loss
    return generation_loss + retrieval_loss_weight * retrieval_loss
