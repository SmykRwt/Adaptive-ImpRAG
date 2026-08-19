import torch
import torch.nn.functional as F
import numpy as np

class MultiRetrieverEnsembleSupervisor:
    """
    Improved Retriever Supervision via Multi-Retriever Ensembling (Section 6.7 of Capstone Report).
    Fuses dense similarity, lexical BM25 matching, and answer grounding verification
    to generate robust, low-noise pseudo-label supervision for retriever training.
    """
    def __init__(self, dense_weight=0.45, lexical_weight=0.30, grounding_weight=0.25):
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.grounding_weight = grounding_weight

    def generate_ensemble_targets(self, dense_scores, lexical_scores, answer_grounding_flags, temperature=0.1):
        """
        dense_scores: Tensor of shape [batch_size, num_candidates]
        lexical_scores: Tensor of shape [batch_size, num_candidates]
        answer_grounding_flags: Boolean Tensor of shape [batch_size, num_candidates]
        Returns:
            ensemble_soft_targets: Tensor of shape [batch_size, num_candidates] summing to 1
        """
        # Normalize dense and lexical scores
        dense_norm = F.softmax(dense_scores / 0.05, dim=-1)
        lexical_norm = F.softmax(lexical_scores / 0.1, dim=-1)
        grounding_score = answer_grounding_flags.float()
        grounding_norm = grounding_score / (grounding_score.sum(dim=-1, keepdim=True).clamp(min=1.0))
        
        # Weighted multi-signal fusion
        fused_score = (
            self.dense_weight * dense_norm +
            self.lexical_weight * lexical_norm +
            self.grounding_weight * grounding_norm
        )
        
        ensemble_soft_targets = F.softmax(fused_score / temperature, dim=-1)
        return ensemble_soft_targets
