import re
import math
from collections import Counter
import numpy as np

class AdaptiveSemanticChunker:
    """
    Advanced Semantic Chunker with sentence-boundary awareness and sliding window overlap.
    Preserves document entity context, title headers, and avoids mid-sentence word chopping.
    """
    def __init__(self, target_chunk_size=150, overlap_size=40, min_chunk_size=30):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        
        # Regex to split on sentence boundaries (. ! ? followed by whitespace or end of string)
        self.sentence_regex = re.compile(r'(?<=[.?!])\s+(?=[A-Z0-9"])')

    def split_into_sentences(self, text):
        """Splits a raw text block into individual clean sentences."""
        text = text.strip()
        if not text:
            return []
        sentences = self.sentence_regex.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text, doc_title=None):
        """
        Chunks document text into overlapping semantic windows respecting sentence boundaries.
        Prepends doc_title if available to anchor pronouns and entity references.
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
            
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        
        title_prefix = f"[{doc_title}] " if doc_title else ""
        
        for sentence in sentences:
            words = sentence.split()
            word_len = len(words)
            
            if current_word_count + word_len > self.target_chunk_size and current_chunk_sentences:
                # Emit current chunk
                chunk_str = " ".join(current_chunk_sentences)
                chunks.append(title_prefix + chunk_str)
                
                # Keep sliding window overlap
                overlap_sentences = []
                overlap_words = 0
                for s in reversed(current_chunk_sentences):
                    s_len = len(s.split())
                    if overlap_words + s_len <= self.overlap_size:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_len
                    else:
                        break
                        
                current_chunk_sentences = overlap_sentences + [sentence]
                current_word_count = overlap_words + word_len
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += word_len
                
        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            if chunks and current_word_count < self.min_chunk_size:
                chunks[-1] = chunks[-1] + " " + chunk_str
            else:
                chunks.append(title_prefix + chunk_str)
            
        return chunks if chunks else [title_prefix + text.strip()]

class AdaptiveRelevanceFilter:
    """
    Hybrid Lexical-Dense Relevance Filter and Re-Ranker.
    Combines dense embedding similarity with BM25/keyword overlap to eliminate
    irrelevant distractor passages before KV injection.
    """
    def __init__(self, lexical_weight=0.35, min_relevance_threshold=0.15):
        self.lexical_weight = lexical_weight
        self.min_relevance_threshold = min_relevance_threshold
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an",
            "and", "any", "are", "as", "at", "be", "because", "been", "before",
            "being", "below", "between", "both", "but", "by", "can", "did", "do",
            "does", "doing", "down", "during", "each", "few", "for", "from",
            "further", "had", "has", "have", "having", "he", "her", "here",
            "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
            "into", "is", "it", "its", "itself", "just", "me", "more", "most",
            "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
            "once", "only", "or", "other", "our", "ours", "ourselves", "out",
            "over", "own", "same", "she", "should", "so", "some", "such", "than",
            "that", "the", "their", "theirs", "them", "themselves", "then",
            "there", "these", "they", "this", "those", "through", "to", "too",
            "under", "until", "up", "very", "was", "we", "were", "what", "when",
            "where", "which", "while", "who", "whom", "why", "with", "would",
            "you", "your", "yours", "yourself", "yourselves"
        }

    def tokenize(self, text):
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def compute_lexical_score(self, query_tokens, passage_text):
        passage_tokens = self.tokenize(passage_text)
        if not passage_tokens or not query_tokens:
            return 0.0
        p_counts = Counter(passage_tokens)
        q_counts = Counter(query_tokens)
        
        # BM25-style term saturation score
        k1 = 1.2
        score = 0.0
        for token in q_counts:
            if token in p_counts:
                tf = p_counts[token]
                score += (tf * (k1 + 1)) / (tf + k1)
        return float(score / (len(query_tokens) + 1e-6))

    def filter_and_rerank(self, query_text, candidate_passages, dense_scores):
        """
        Re-ranks passages using hybrid dense + lexical scoring and filters out low-scoring distractors.
        Returns:
            ranked_passages: list of filtered and re-ranked passage strings
            ranked_scores: list of final hybrid scores
        """
        q_tokens = self.tokenize(query_text)
        
        scored_items = []
        for i, passage in enumerate(candidate_passages):
            dense_s = float(dense_scores[i]) if i < len(dense_scores) else 0.0
            lexical_s = self.compute_lexical_score(q_tokens, passage)
            
            # Normalize dense score assuming typical FAISS dot product in [-1, 1]
            norm_dense = max(0.0, min(1.0, (dense_s + 1.0) / 2.0))
            norm_lexical = min(1.0, lexical_s)
            
            hybrid_score = (1.0 - self.lexical_weight) * norm_dense + self.lexical_weight * norm_lexical
            
            # Keep if passes minimum threshold or is top candidate
            if hybrid_score >= self.min_relevance_threshold or i == 0:
                scored_items.append((passage, hybrid_score, dense_s, lexical_s))
                
        # Sort descending by hybrid score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        filtered_passages = [item[0] for item in scored_items]
        filtered_scores = [item[1] for item in scored_items]
        
        return filtered_passages, filtered_scores
