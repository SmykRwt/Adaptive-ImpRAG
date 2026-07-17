import random
import torch
from torch.utils.data import Dataset

# Prompt templates from Table 1 and Table 9
TEMPLATES = {
    "nq": "Q: {question} A: {answer}",
    "hopo": "Q: {question} A: {answer}",
    "sqa": "Q: {question} A: {answer}",
    "2wqa": "Q: {question} A: {answer}",
    "aida": "{context} Output the Wikipedia page title of the entity mentioned between [START] and [END] in the given text A: {answer}",
    "fev": "Is this statement true? {statement} A: {answer}",
    "t_rex": "{entity} [SEP] {relation} Provide the answer corresponding to the relation specified after [SEP] for the entity mentioned before [SEP] A: {answer}",
    "zsre": "{entity} [SEP] {relation} Provide the answer corresponding to the relation specified after [SEP] for the entity mentioned before [SEP] A: {answer}",
    "dialogue": "{context}",
    "reading_comp": "{context} Q: {question} A: {answer}",
    "summarization": "{context} Summarize this article: {answer}",
    "phrase_denoising": "{context} Recover the original phrases marked between [START] and [END] in the given text A: {answer}",
    "sentence_gen": "{context} [SEP] {direction} sentence Generate a sentence corresponding to the relation specified after [SEP] for the context mentioned before [SEP] A: {answer}"
}

def format_prompt(task_type, **kwargs):
    """
    Formats the prompt according to the task type templates.
    """
    if task_type not in TEMPLATES:
        raise ValueError(f"Unknown task type: {task_type}")
    return TEMPLATES[task_type].format(**kwargs)

class SyntheticTaskGenerator:
    """
    Helper to generate synthetic tasks:
    1. Phrase Denoising
    2. Next/Previous Sentence Generation
    """
    @staticmethod
    def generate_phrase_denoising(text):
        """
        Takes a paragraph, inserts [START] and [END] around a random phrase,
        and returns the context (prompt) and target phrase (answer).
        """
        words = text.split()
        if len(words) < 10:
            return text, ""
            
        phrase_len = random.randint(2, 5)
        start_idx = random.randint(2, len(words) - phrase_len - 2)
        end_idx = start_idx + phrase_len
        
        phrase = " ".join(words[start_idx:end_idx])
        
        words_with_tags = words[:start_idx] + ["[START]"] + words[start_idx:end_idx] + ["[END]"] + words[end_idx:]
        context = " ".join(words_with_tags)
        
        prompt = format_prompt("phrase_denoising", context=context, answer="")
        # Remove trailing "A: " for prompt-only encoding, or return full pair
        return prompt, phrase

    @staticmethod
    def generate_sentence_gen(text):
        """
        Splits a paragraph into sentences, picks consecutive ones,
        and returns the first sentence, direction, and target sentence.
        """
        # A simple sentence splitter using punctuation
        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if len(s.strip()) > 10]
        if len(sentences) < 2:
            return text, ""
            
        idx = random.randint(0, len(sentences) - 2)
        direction = random.choice(["next", "previous"])
        
        if direction == "next":
            context = sentences[idx]
            answer = sentences[idx + 1]
        else:
            context = sentences[idx + 1]
            answer = sentences[idx]
            
        prompt = format_prompt("sentence_gen", context=context, direction=direction, answer="")
        return prompt, answer

class ImpRAGDataset(Dataset):
    """
    PyTorch Dataset class for ImpRAG.
    Each sample contains:
    - query: string query/prompt
    - answer: string answer/response
    - positive_passages: list of positive passage strings
    - negative_passages: list of negative passage strings (hard + random)
    """
    def __init__(self, data_list):
        """
        data_list: list of dicts. Each dict must contain:
          - "query": str
          - "answer": str
          - "positive_passages": list of str (optional)
          - "negative_passages": list of str (optional)
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "answer": item["answer"],
            "positive_passages": item.get("positive_passages", []),
            "negative_passages": item.get("negative_passages", [])
        }

def collate_fn(batch, tokenizer, max_query_len=128, max_passage_len=32, num_candidates=5):
    """
    Collation function to prepare batch tensors.
    """
    queries = [item["query"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    # Tokenize query prompts and answers
    tokenized_queries = tokenizer(queries, padding=True, truncation=True, max_length=max_query_len, return_tensors="pt")
    
    # Tokenize query + answer sequences for causal language modeling loss
    full_sequences = [f"{q} {a}" for q, a in zip(queries, answers)]
    tokenized_full = tokenizer(full_sequences, padding=True, truncation=True, max_length=max_query_len + 32, return_tensors="pt")
    
    # Create labels where all prompt tokens are marked as -100 (ignored in loss computation)
    labels = tokenized_full.input_ids.clone()
    for i in range(len(batch)):
        q_len = len(tokenizer.encode(queries[i]))
        labels[i, :q_len] = -100
    
    # Process passages for retrieval warmup and distillation
    # We want a candidate set of size `num_candidates` per query
    batch_pos_passages = []
    batch_neg_passages = []
    positive_mask = []
    
    for item in batch:
        pos = item["positive_passages"]
        neg = item["negative_passages"]
        
        # Ensure we have at least one positive
        if len(pos) == 0:
            pos = ["Dummy positive passage."]
        if len(neg) == 0:
            neg = ["Dummy negative passage."]
            
        # Select one positive and (num_candidates - 1) negatives
        selected_pos = random.choice(pos)
        
        selected_negs = []
        if len(neg) >= num_candidates - 1:
            selected_negs = random.sample(neg, num_candidates - 1)
        else:
            # Pad negatives if not enough
            selected_negs = neg + [random.choice(neg) for _ in range(num_candidates - 1 - len(neg))]
            
        candidates = [selected_pos] + selected_negs
        batch_pos_passages.append(selected_pos)
        batch_neg_passages.extend(selected_negs)
        
        # In candidates list [pos, neg_1, neg_2, ...], the first one is positive
        mask = [True] + [False] * (num_candidates - 1)
        positive_mask.append(mask)
        
    # Tokenize candidate passages
    all_candidates = []
    for i in range(len(batch)):
        pos_passage = batch_pos_passages[i]
        neg_passages = batch_neg_passages[i * (num_candidates - 1) : (i + 1) * (num_candidates - 1)]
        all_candidates.append(pos_passage)
        all_candidates.extend(neg_passages)
        
    tokenized_candidates = tokenizer(all_candidates, padding=True, truncation=True, max_length=max_passage_len, return_tensors="pt")
    
    return {
        "query_ids": tokenized_queries.input_ids,
        "query_attention_mask": tokenized_queries.attention_mask,
        "full_ids": tokenized_full.input_ids,
        "full_attention_mask": tokenized_full.attention_mask,
        "labels": labels,
        "candidate_passage_ids": tokenized_candidates.input_ids,
        "candidate_passage_attention_mask": tokenized_candidates.attention_mask,
        "positive_mask": torch.tensor(positive_mask, dtype=torch.bool)
    }
