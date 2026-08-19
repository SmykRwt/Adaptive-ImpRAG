import re
import numpy as np
from collections import Counter

class DocumentUtilityScorer:
    """
    Document Utility Scoring & Sufficiency Module (Section 6.4 of Capstone Report).
    Evaluates each retrieved passage for semantic relevance, information novelty,
    and factual sufficiency to filter out noise before KV injection.
    """
    def __init__(self, utility_threshold=0.20, redundancy_penalty=0.30):
        self.utility_threshold = utility_threshold
        self.redundancy_penalty = redundancy_penalty
        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "is", "are", "was", "were", "be", "been",
            "to", "of", "in", "on", "for", "with", "by", "at", "from", "it",
            "its", "he", "she", "they", "their", "his", "her", "who", "whom"
        }

    def tokenize(self, text):
        return [w for w in re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()) if w not in self.stop_words and len(w) > 1]

    def compute_jaccard_similarity(self, tokens_a, tokens_b):
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def score_and_filter_passages(self, query_text, candidate_passages, candidate_scores=None):
        """
        Scores candidate passages on:
        1. Relevance utility U_rel(p, q)
        2. Information novelty U_nov(p_i, P_selected)
        Returns:
            selected_passages: prioritized list of high-utility non-redundant passages
            utility_report: dict containing utility scores and filtering stats
        """
        q_tokens = self.tokenize(query_text)
        if candidate_scores is None:
            candidate_scores = [1.0] * len(candidate_passages)
            
        selected_passages = []
        selected_tokens_list = []
        utility_scores = []
        
        for i, passage in enumerate(candidate_passages):
            p_tokens = self.tokenize(passage)
            if not p_tokens:
                continue
                
            # 1. Query relevance (Keyword overlap + dense score)
            overlap_count = sum(1 for t in q_tokens if t in p_tokens)
            lexical_rel = overlap_count / max(1, len(q_tokens))
            dense_s = float(candidate_scores[i]) if i < len(candidate_scores) else 0.5
            norm_dense = max(0.0, min(1.0, (dense_s + 1.0) / 2.0))
            
            raw_relevance = 0.5 * norm_dense + 0.5 * lexical_rel
            
            # 2. Redundancy / Novelty check against already selected passages
            max_redundancy = 0.0
            for prev_tokens in selected_tokens_list:
                sim = self.compute_jaccard_similarity(p_tokens, prev_tokens)
                if sim > max_redundancy:
                    max_redundancy = sim
                    
            # Utility = Relevance - Redundancy Penalty
            utility = raw_relevance - (self.redundancy_penalty * max_redundancy)
            
            if utility >= self.utility_threshold or len(selected_passages) == 0:
                selected_passages.append(passage)
                selected_tokens_list.append(p_tokens)
                utility_scores.append(float(utility))
                
        # Calculate context sufficiency indicator
        total_unique_terms_covered = len(set(t for tokens in selected_tokens_list for t in tokens if t in q_tokens))
        coverage_ratio = total_unique_terms_covered / max(1, len(set(q_tokens)))
        is_context_sufficient = coverage_ratio >= 0.60
        
        return selected_passages, {
            "utility_scores": utility_scores,
            "coverage_ratio": float(coverage_ratio),
            "is_context_sufficient": bool(is_context_sufficient),
            "num_filtered_out": len(candidate_passages) - len(selected_passages)
        }
