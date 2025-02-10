import pandas as pd
import os

from llm import llm_chat

# Define CSV filename
csv_filename = "llm_responses_baseline.csv"

# Initialize results list and determine starting index
results = []

answer_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
df = pd.read_parquet("hf://datasets/guan-wang/ReDis-QA/data/test-00000-of-00001.parquet")

for idx in range(100):
    row = df.iloc[idx]
    prompt = f"""Please answer the following multiple choice question by selecting ONLY the letter A, B, C, or D. Do not include any other text in your answer choice.

Question: {row['input']}

You must choose exactly one answer from A, B, C, or D."""
    response = llm_chat(prompt, 'answer')
    answer = response['chosen_answer']
    rationale = response['explanation']

    # Get the correct answer
    correct_answer = answer_dict.get(row['cop'], 'Unknown')

    # Store the result
    result = {
        'index': idx,
        'question': row['input'],
        'llm_answer': answer,
        'correct_answer': correct_answer,
        'rationale': rationale
    }
    print(f'{idx}: Question: {row["input"]}\nLLM Answer: {answer}\nCorrect Answer: {correct_answer}\nRationale: {rationale}')
    results.append(result)

# Save to CSV after each response
results_df = pd.DataFrame(results)
results_df.to_csv(csv_filename, index=False)

# Calculate and print accuracy
accuracy = results_df[results_df['llm_answer'] == results_df['correct_answer']].shape[0] / results_df.shape[0]
print(f"Accuracy: {accuracy:.2%}")