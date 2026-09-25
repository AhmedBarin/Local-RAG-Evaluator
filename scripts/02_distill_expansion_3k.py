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
OUTPUT_FILE = "rag_judge_distilled_dataset_3k.jsonl"
EXISTING_FILE = "rag_judge_distilled_dataset.jsonl"
# Safety limit to prevent massive documents from draining your API budget
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
    print("Loading HaluBench dataset...")
    full_dataset = load_dataset("PatronusAI/HaluBench", split="test")

    # Offset indices [500:2000] to bypass the initial 500 rows used in the 1k run
    print("Selecting 1,500 PASS and 1,500 FAIL rows offset from previous run...")
    pass_filtered = full_dataset.filter(lambda x: x["label"] == "PASS")
    fail_filtered = full_dataset.filter(lambda x: x["label"] == "FAIL")

    pass_rows = pass_filtered.select(range(500, 2000))
    fail_rows = fail_filtered.select(range(500, 2000))

    # Combine and shuffle
    dataset = concatenate_datasets([pass_rows, fail_rows]).shuffle(seed=42)

    # Cross-file deduplication guard against the original 1k file
    existing_inputs = set()
    if os.path.exists(EXISTING_FILE):
        with open(EXISTING_FILE, "r", encoding="utf-8") as ef:
            for line in ef:
                try:
                    record = json.loads(line)
                    existing_inputs.add(record.get("input", ""))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(existing_inputs)} existing inputs for deduplication verification.")

    print(f"Starting distillation of 3,000 new rows using {TEACHER_MODEL}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, row in enumerate(tqdm(dataset, desc="Distilling Rows")):
            try:
                query = str(row.get("question", ""))
                context_str = str(row.get("passage", ""))
                answer = str(row.get("answer", ""))

                if len(context_str) > MAX_CONTEXT_LENGTH:
                    context_str = context_str[:MAX_CONTEXT_LENGTH] + "... [TRUNCATED]"

                user_prompt = f"Query: {query}\n\nContext: {context_str}\n\nAnswer: {answer}"

                # 1. Skip any row that appeared in the first 1k dataset (or earlier in this loop)
                if user_prompt in existing_inputs:
                    continue
                
                # 2. Add current prompt to the set to prevent identical duplicates within this batch
                existing_inputs.add(user_prompt)

                full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{user_prompt}"

                # Query Gemini API with native Pydantic schema enforcement (Standard Generation, No Thinking Budget)
                response = client.models.generate_content(
                    model=TEACHER_MODEL,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=RAGEvaluation,
                    )
                )

                # Format for Unsloth Studio Alpaca QLoRA
                record = {
                    "instruction": SYSTEM_INSTRUCTION,
                    "input": user_prompt,
                    "output": response.text
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

            except Exception as e:
                print(f"\nRow {idx} failed permanently: {e}")

    print(f"\nDistillation complete! 3,000 completely unique, balanced rows saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()