import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prepare Wikipedia Corpus")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="wikipedia_simple", 
        choices=["wikitext-2", "wikitext-103", "wikipedia_simple"],
        help="Wikipedia dataset configuration to use"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Wikipedia Corpus Preparation (using {args.dataset})")
    print("=" * 60)
    
    passages = []
    
    if args.dataset == "wikipedia_simple":
        print("Downloading simple Wikipedia dataset (20220301.simple) from Hugging Face...")
        # Simple Wikipedia has pre-tokenized paragraphs or full texts
        dataset = load_dataset("wikipedia", "20220301.simple", split="train")
        
        print("Processing and chunking articles into 128-word passages...")
        for item in tqdm(dataset, desc="Processing articles"):
            text = item["text"].strip()
            if not text:
                continue
            words = text.split()
            current_chunk = []
            word_count = 0
            for word in words:
                current_chunk.append(word)
                word_count += 1
                if word_count >= 120:  # 120 words target for a ~128 token max budget
                    passages.append(" ".join(current_chunk))
                    current_chunk = []
                    word_count = 0
            if current_chunk:
                passages.append(" ".join(current_chunk))
                
    else:
        config = "wikitext-2-raw-v1" if args.dataset == "wikitext-2" else "wikitext-103-raw-v1"
        print(f"Downloading {config} dataset from Hugging Face...")
        dataset = load_dataset("Salesforce/wikitext", config, split="train")
        
        print("Processing and chunking articles into 128-word passages...")
        current_chunk = []
        word_count = 0
        for item in tqdm(dataset, desc="Processing rows"):
            text = item["text"].strip()
            if not text:
                continue
            # Skip section headers
            if text.startswith("=") and text.endswith("="):
                continue
                
            words = text.split()
            for word in words:
                current_chunk.append(word)
                word_count += 1
                if word_count >= 120:
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
