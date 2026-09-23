import os
import json
from tqdm import tqdm
from pydantic import BaseModel, Field
from datasets import load_dataset, concatenate_datasets
from google import genai
from google.genai import types

# Initialize Client with Auto-Retry for Rate Limits
client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            initial_delay=2.0,
            attempts=5,
            max_delay=60.0,
            http_status_codes=[429, 500, 502, 503, 504]
        ),
        timeout=120000,
    )
)

TEACHER_MODEL = "gemini-3.8-flash"
OUTPUT_FILE = "rag_judge_distilled_dataset.jsonl"
# Safety limit to prevent massive FinanceBench documents from draining your API budget
MAX_CONTEXT_LENGTH = 6000 

# Define the Strict JSON Schema
class MetricScore(BaseModel):
    score: int = Field(..., description="Integer score between 1 and 5")
    reasoning: str = Field(..., description="Short explanation justifying the assigned score")

class RAGEvaluation(BaseModel):
    context_relevance: MetricScore
    groundedness: MetricScore
    answer_relevance: MetricScore
    hallucination_flag: bool = Field(..., description="True if answer introduces ungrounded factual claims")

SYSTEM_INSTRUCTION = (
    "You are a strict RAG evaluator. Analyze the user Query, the Retrieved Context, "
    "and the Generated Answer. Output your evaluation in strict JSON format scoring "
    "Context Relevance, Groundedness, and Answer Relevance from 1 to 5, along with a "
    "hallucination flag and concise reasoning for each metric."
)

def main():
    print("Loading and balancing HaluBench dataset...")
    
    # HaluBench uses the 'test' split, not 'train'
    full_dataset = load_dataset("PatronusAI/HaluBench", split="test")
    
    print("Filtering 50/50 Pass/Fail split (500 rows each)...")
    # Force a perfect 50/50 balance (500 truthful + 500 hallucinations = 1,000 rows)
    pass_rows = full_dataset.filter(lambda x: x["label"] == "PASS").select(range(500))
    fail_rows = full_dataset.filter(lambda x: x["label"] == "FAIL").select(range(500))
    
    # Combine and shuffle so the model doesn't learn a sequential pattern
    dataset = concatenate_datasets([pass_rows, fail_rows]).shuffle(seed=42)
    
    print(f"Starting generation of 1,000 rows using {TEACHER_MODEL}...")
    
    # Using 'w' mode to start completely fresh
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, row in enumerate(tqdm(dataset)):
            try:
                # Extract and safely cast to string to prevent null/integer TypeErrors
                query = str(row.get("question", ""))
                context_str = str(row.get("passage", ""))
                answer = str(row.get("answer", ""))
                
                # Truncate context to protect the API budget
                if len(context_str) > MAX_CONTEXT_LENGTH:
                    context_str = context_str[:MAX_CONTEXT_LENGTH] + "... [TRUNCATED]"

                user_prompt = f"Query: {query}\n\nContext: {context_str}\n\nAnswer: {answer}"
                full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{user_prompt}"

                # Query Gemini API with native Pydantic enforcement
                response = client.models.generate_content(
                    model=TEACHER_MODEL,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=RAGEvaluation,
                    )
                )

                # Format for Unsloth Studio QLoRA
                record = {
                    "instruction": SYSTEM_INSTRUCTION,
                    "input": user_prompt,
                    "output": response.text
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

            except Exception as e:
                print(f"\nRow {idx} failed permanently: {e}")

    print(f"\nGeneration complete! Perfectly balanced 1k dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()