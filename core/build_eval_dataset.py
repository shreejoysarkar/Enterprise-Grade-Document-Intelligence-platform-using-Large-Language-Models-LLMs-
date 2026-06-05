# generate_manual_eval_csv.py

import pandas as pd

from core.retrieval_and_generation_4 import RAGPipeline
from evaluation_dataset import evaluation_data

print("Initializing RAG Pipeline...")

pipeline = RAGPipeline()

rows = []

for idx, item in enumerate(evaluation_data, start=1):

    question = item["question"]
    ground_truth = item["ground_truth"]

    print(f"Processing Question {idx}/{len(evaluation_data)}")

    # Retrieve context
    contexts = pipeline.retrieve_and_rerank(
        question,
        retrieve_top_n=20,
        keep_top_k=4
    )

    # Generate answer
    answer = pipeline.generate(
        question,
        stream=False
    )

    # Convert retrieved chunks into readable text
    retrieved_context = "\n\n".join(
        chunk["metadata"].get("text", "")
        for chunk in contexts
    )

    rows.append({
    "question": question,
    "ground_truth": ground_truth,
    "generated_answer": answer,

    "context_1": contexts[0]["metadata"].get("text", "")
                 if len(contexts) > 0 else "",

    "context_2": contexts[1]["metadata"].get("text", "")
                 if len(contexts) > 1 else "",

    "context_3": contexts[2]["metadata"].get("text", "")
                 if len(contexts) > 2 else "",

    "context_4": contexts[3]["metadata"].get("text", "")
                 if len(contexts) > 3 else ""
    })

# Save CSV
df = pd.DataFrame(rows)

df.to_csv(
    "evaluation_results.csv",
    index=False,
    encoding="utf-8"
)

print("CSV created successfully!")
print("Saved as evaluation_results.csv")