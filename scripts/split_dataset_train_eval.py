import random

def split_dataset(input_file="combined_dataset_3k.jsonl", train_file="train_split.jsonl", eval_file="eval_split.jsonl", train_ratio=0.9):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Shuffle predictably to ensure an even mix of True/False hallucinations in both sets
    random.seed(42)
    random.shuffle(lines)
    
    split_index = int(len(lines) * train_ratio)
    train_lines = lines[:split_index]
    eval_lines = lines[split_index:]
    
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
        
    with open(eval_file, "w", encoding="utf-8") as f:
        f.writelines(eval_lines)
        
    print(f"✅ Split complete!")
    print(f"Training set: {len(train_lines)} rows saved to '{train_file}'")
    print(f"Evaluation set: {len(eval_lines)} rows saved to '{eval_file}'")

if __name__ == "__main__":
    split_dataset()