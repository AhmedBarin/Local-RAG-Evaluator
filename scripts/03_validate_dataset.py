import json
import random
import sys
from collections import Counter

def validate_dataset(filepath="combined_dataset_4k.jsonl"):
    valid_rows = 0
    failed_rows = 0
    unique_inputs = set()
    
    # Track distributions across all Pydantic schema metrics
    context_relevance_scores = Counter()
    groundedness_scores = Counter()
    answer_relevance_scores = Counter()
    hallucinations = Counter()
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Check your current directory.")
        return

    print(f"Total Rows in File: {len(lines)}")
    
    for i, line in enumerate(lines):
        try:
            # 1. Verify Unsloth Alpaca template structure
            data = json.loads(line.strip())
            if not all(k in data for k in ["instruction", "input", "output"]):
                raise KeyError("Missing required Alpaca format keys (instruction, input, output)")
            
            # 2. Track unique inputs for duplication monitoring
            unique_inputs.add(data["input"])
            
            # 3. Verify Pydantic schema keys and nested structure
            output_json = json.loads(data["output"])
            required_metrics = ["context_relevance", "groundedness", "answer_relevance", "hallucination_flag"]
            if not all(k in output_json for k in required_metrics):
                 raise KeyError("Missing required Pydantic metric keys in the output payload")
            
            # 4. Tally dataset balance across all scores
            context_relevance_scores[output_json["context_relevance"]["score"]] += 1
            groundedness_scores[output_json["groundedness"]["score"]] += 1
            answer_relevance_scores[output_json["answer_relevance"]["score"]] += 1
            hallucinations[output_json["hallucination_flag"]] += 1
            
            valid_rows += 1
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            failed_rows += 1
            print(f"Row {i} is corrupted: {e}")
            
    # Calculate duplication metrics based on unique queries/passages
    duplicate_count = valid_rows - len(unique_inputs)

    # Print Comprehensive Report
    print("\n================ DATASET REPORT ================")
    print(f"File Checked:       {filepath}")
    print(f"Total Lines:        {len(lines)}")
    print(f"Valid JSON Rows:    {valid_rows}")
    print(f"Corrupted Rows:     {failed_rows}")
    print(f"Unique Inputs:      {len(unique_inputs)}")
    print(f"Duplicate Prompts:  {duplicate_count}")
    
    def print_distribution(counter_obj, name):
        print(f"\n--- {name} ---")
        for key, count in sorted(counter_obj.items()):
            percentage = (count / valid_rows) * 100 if valid_rows > 0 else 0
            print(f"  {key}: {count} ({percentage:.1f}%)")

    print_distribution(context_relevance_scores, "Context Relevance Scores (1-5)")
    print_distribution(groundedness_scores, "Groundedness Scores (1-5)")
    print_distribution(answer_relevance_scores, "Answer Relevance Scores (1-5)")
    print_distribution(hallucinations, "Hallucination Flags (True/False)")
    
    print("================================================\n")

    if valid_rows > 0:
        print("--- Random Output Sample ---")
        # Ensure we only pick a sample from structurally valid rows
        sample_row = json.loads(random.choice([line for line in lines if 'hallucination_flag' in line]))
        print(json.dumps(json.loads(sample_row["output"]), indent=2))

if __name__ == "__main__":
    # Allows you to pass the file name directly in the terminal, defaulting to the merged file
    target_file = sys.argv[1] if len(sys.argv) > 1 else "combined_dataset_4k.jsonl"
    validate_dataset(target_file)