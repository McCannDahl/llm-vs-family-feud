import csv
import time
from collections import Counter
import numpy as np
import ollama
import pandas as pd

NUM_RUNS = 100
SIMILARITY_THRESHOLD = 0.5  # Change as needed
ERROR_KEYWORDS = ["i don't know", "not sure", "can't answer", "cannot answer", "uncertain"]

# --- LLM interaction ---
def ask_llm(question: str, model_name: str) -> str:
    response = ollama.chat(
        messages=[
            {
                'role': 'system',
                'content': 'You are an AI assistant that answers questions. Only reply with one answer. Do not reply with a list. Your reply should be short, typically two words or a short phrase.',
            },
            {
                'role': 'user',
                'content': question,
            }
        ],
        options={"temperature": 1, 'top_k': 0},
        model=model_name,
    )
    return response.message.content.strip() if response.message.content else ""

def get_embedding(text: str) -> np.ndarray:
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return np.array(response['embedding'])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar_answer(response: str, candidates: list[str]) -> tuple:
    if any(error_phrase in response.lower() for error_phrase in ERROR_KEYWORDS):
        return "error", 0

    response_emb = get_embedding(response)
    best_match = None
    best_score = -1

    for candidate in candidates:
        candidate_emb = get_embedding(candidate)
        similarity = cosine_similarity(response_emb, candidate_emb)
        if similarity > best_score:
            best_match = candidate
            best_score = similarity

    if best_score >= SIMILARITY_THRESHOLD and best_match is not None:
        return best_match, best_score
    return "none", 0

# --- Simulation ---
def run_simulation(question: str, candidate_answers: list[str], model_name: str) -> Counter:
    counter = Counter()
    for i in range(NUM_RUNS):
        print(f"🧠 [{model_name}] Q: {question} (#{i+1})")
        response = ask_llm(question, model_name)
        print(f"LLM: {response}")
        match, best_score = find_most_similar_answer(response, candidate_answers)
        print(f"Matched to: {match} best_score={best_score}\n")
        counter[match] += 1
        time.sleep(0.2)  # Small delay
    return counter

# --- Main ---
def load_input_csv(path: str) -> list[dict]:
    df = pd.read_csv(path)
    questions = []

    for _, row in df.iterrows():
        question = row["Question"]
        candidates = []
        for i in range(1, 9):
            answer_col = f"Answer {i}"
            if pd.notna(row.get(answer_col)):
                candidates.append(str(row[answer_col]).strip())
        questions.append({
            "question": question,
            "candidates": candidates,
            "row_data": row.to_dict()
        })
    return questions

def save_output_csv(results: list[dict], model_names: list[str], output_path: str):
    rows = []
    for item in results:
        row_data = item["row_data"]
        all_counters: dict[str, Counter] = item["counters"]

        for model_name in model_names:
            counter = all_counters[model_name]
            for i in range(1, 10):
                answer_col = f"Answer {i}"
                llm_col = f"{answer_col} {model_name} likelihood"
                answer = str(row_data.get(answer_col, '')).strip()
                if answer:
                    row_data[llm_col] = counter[answer]
                else:
                    row_data[llm_col] = ""
            # Add "none" and "error" counts
            row_data[f"none {model_name} likelihood"] = counter["none"]
            row_data[f"error {model_name} likelihood"] = counter["error"]

        rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == "__main__":
    print("🎉 Starting Multi-LLM Family Feud Simulation...\n")
    input_path = "input.csv"
    output_path = "output.csv"

    model_names = ["llama3.2", "gemma3:4b-it-qat"]  # Add more models as needed
    questions = load_input_csv(input_path)

    # Initialize results as a map of question ID -> result object
    question_results = {i: {
        "row_data": q["row_data"],
        "counters": {}
    } for i, q in enumerate(questions)}

    for model_name in model_names:
        print(f"\n🚀 Running model: {model_name}")
        for i, q in enumerate(questions):
            print(f"\n🔍 [{model_name}] Question: {q['question']}")
            counter = run_simulation(q['question'], q['candidates'], model_name)
            question_results[i]["counters"][model_name] = counter

    # Reassemble the results list
    results = [question_results[i] for i in range(len(questions))]
    save_output_csv(results, model_names, output_path)

