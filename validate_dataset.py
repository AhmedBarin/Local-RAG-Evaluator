import json
import random
from collections import Counter

def validate_dataset(filepath="rag_judge_distilled_dataset.jsonl"):
    valid_rows = 0
    failed_rows = 0
    groundedness_scores = Counter()
    hallucinations = Counter()
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"Total Rows Generated: {len(lines)}")
    
    for i, line in enumerate(lines):
        try:
            # 1. Verify Unsloth Alpaca template structure
            data = json.loads(line)
            if not all(k in data for k in ["instruction", "input", "output"]):
                raise KeyError("Missing required Alpaca format keys")
            
            # 2. Verify Pydantic schema adherence
            output_json = json.loads(data["output"])
            
            # 3. Tally dataset balance
            groundedness_scores[output_json["groundedness"]["score"]] += 1
            hallucinations[output_json["hallucination_flag"]] += 1
            valid_rows += 1
            
        except (json.JSONDecodeError, KeyError) as e:
            failed_rows += 1
            print(f"Row {i} is corrupted: {e}")
            
    print(f"Valid Rows: {valid_rows}")
    print(f"Corrupted Rows: {failed_rows}")
    
    print("\n--- Score Distribution ---")
    print(f"Groundedness Scores (1-5): {dict(sorted(groundedness_scores.items()))}")
    print(f"Hallucination Flags: {dict(hallucinations)}")

    if valid_rows > 0:
        print("\n--- Random Output Sample ---")
        sample_row = json.loads(random.choice(lines))
        print(json.dumps(json.loads(sample_row["output"]), indent=2))

if __name__ == "__main__":
    validate_dataset()