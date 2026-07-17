import json
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("=" * 60)
    print("Wikipedia Corpus Preparation (using WikiText-2)")
    print("=" * 60)
    
    print("Downloading wikitext-2-raw-v1 dataset from Hugging Face...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    
    print("Processing and chunking articles into 128-word passages...")
    passages = []
    current_chunk = []
    word_count = 0
    
    for item in tqdm(dataset, desc="Processing rows"):
        text = item["text"].strip()
        if not text:
            continue
        # Skip section headers (e.g., "= Heading =")
        if text.startswith("=") and text.endswith("="):
            continue
            
        words = text.split()
        for word in words:
            current_chunk.append(word)
            word_count += 1
            if word_count >= 120:  # 120 words target for a ~128 token max budget
                passages.append(" ".join(current_chunk))
                current_chunk = []
                word_count = 0
                
    if current_chunk:
        passages.append(" ".join(current_chunk))
        
    print(f"Generated {len(passages)} passages.")
    
    # Save the corpus
    output_path = "wiki_passages.json"
    print(f"Saving corpus to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
        
    print("Corpus preparation complete!")

if __name__ == "__main__":
    main()
