import pandas as pd

df = pd.read_json("rag_eval_dataset.json")

hallucinated = 0

for _, row in df.iterrows():

    answer = row["answer"].lower()

    context = " ".join(
        row["contexts"]
    ).lower()

    if answer not in context:
        hallucinated += 1

rate = hallucinated / len(df)

print(
    f"Hallucination Rate: {rate:.2%}"
)