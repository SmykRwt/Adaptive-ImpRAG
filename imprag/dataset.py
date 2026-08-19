import random
import torch
from torch.utils.data import Dataset

# Prompt templates from Table 1 and Table 9 in the ImpRAG paper
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
    Helper to generate synthetic tasks (Section 4.1):
    1. Phrase Denoising
    2. Next/Previous Sentence Generation
    """
    @staticmethod
    def generate_phrase_denoising(text):
        words = text.split()
        if len(words) < 10:
            return text, ""
            
        phrase_len = random.randint(2, 5)
        start_idx = random.randint(2, max(2, len(words) - phrase_len - 2))
        end_idx = min(len(words), start_idx + phrase_len)
        
        phrase = " ".join(words[start_idx:end_idx])
        
        words_with_tags = words[:start_idx] + ["[START]"] + words[start_idx:end_idx] + ["[END]"] + words[end_idx:]
        context = " ".join(words_with_tags)
        
        prompt = format_prompt("phrase_denoising", context=context, answer="")
        return prompt, phrase

    @staticmethod
    def generate_sentence_gen(text):
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
    - positive_passages: list of positive passage strings (P(q))
    - negative_passages: list of hard negative passage strings (Nh(q))
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "answer": item["answer"],
            "positive_passages": item.get("positive_passages", item.get("pos", [])),
            "negative_passages": item.get("negative_passages", item.get("negs", []))
        }

def collate_fn(batch, tokenizer, max_query_len=128, max_passage_len=128, max_positives_per_query=5, max_negatives_per_query=5):
    """
    Collation function preparing:
    1. Query tokens and attention mask.
    2. Full query + response sequences and labels with -100 masking on query tokens.
    3. Candidate passage pool: includes all pseudo-positives P(q) and hard negatives Nh(q) across batch items.
    4. positive_mask: [batch_size, num_total_candidates] boolean matrix identifying positive candidates for each query.
    5. primary_positive_ids: primary positive passage tokens for reader generation training.
    """
    batch_size = len(batch)
    queries = [item["query"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    # 1. Tokenize query prompts
    tokenized_queries = tokenizer(queries, padding=True, truncation=True, max_length=max_query_len, return_tensors="pt")
    
    # 2. Tokenize full sequences (query + answer) for causal language modeling
    full_sequences = [f"{q.strip()} {a.strip()}" for q, a in zip(queries, answers)]
    tokenized_full = tokenizer(full_sequences, padding=True, truncation=True, max_length=max_query_len + 64, return_tensors="pt")
    
    # Create labels where all prompt tokens are marked as -100
    labels = tokenized_full.input_ids.clone()
    for i in range(batch_size):
        q_len = len(tokenizer.encode(queries[i].strip(), add_special_tokens=False))
        # Ensure we do not mask the entire sequence
        q_len = min(q_len, labels.shape[1] - 1)
        labels[i, :q_len] = -100
        
    # 3. Assemble Candidate Passages pool: P(q) + Nh(q)
    all_candidate_texts = []
    # Map (batch_idx, candidate_idx) -> bool
    positive_flags = []
    
    primary_positives = []
    
    for i, item in enumerate(batch):
        pos_list = item["positive_passages"]
        neg_list = item["negative_passages"]
        
        # Take primary positive for reader generation
        primary_p = pos_list[0] if len(pos_list) > 0 else "No relevant passage available."
        primary_positives.append(primary_p)
        
        # Add positives
        used_pos = pos_list[:max_positives_per_query] if len(pos_list) > 0 else [primary_p]
        for p in used_pos:
            cand_idx = len(all_candidate_texts)
            all_candidate_texts.append(p)
            positive_flags.append((i, cand_idx))
            
        # Add hard negatives
        used_negs = neg_list[:max_negatives_per_query]
        for n in used_negs:
            all_candidate_texts.append(n)
            
    # Tokenize candidate passages
    tokenized_candidates = tokenizer(
        all_candidate_texts,
        padding=True,
        truncation=True,
        max_length=max_passage_len,
        return_tensors="pt"
    )
    
    # Tokenize primary positives for generator loss
    tokenized_primary_pos = tokenizer(
        primary_positives,
        padding=True,
        truncation=True,
        max_length=max_passage_len,
        return_tensors="pt"
    )
    
    num_candidates = len(all_candidate_texts)
    positive_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool)
    for q_idx, c_idx in positive_flags:
        positive_mask[q_idx, c_idx] = True
        
    return {
        "query_ids": tokenized_queries.input_ids,
        "query_attention_mask": tokenized_queries.attention_mask,
        "full_ids": tokenized_full.input_ids,
        "full_attention_mask": tokenized_full.attention_mask,
        "labels": labels,
        "candidate_passage_ids": tokenized_candidates.input_ids,
        "candidate_passage_attention_mask": tokenized_candidates.attention_mask,
        "positive_mask": positive_mask,
        "primary_positive_ids": tokenized_primary_pos.input_ids,
        "primary_positive_attention_mask": tokenized_primary_pos.attention_mask
    }
