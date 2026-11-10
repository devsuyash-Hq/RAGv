import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={
        "model": EMBED_MODEL,
        "input": texts
    })
    return np.array(r.json()["embeddings"])


def ask_llm(prompt):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    })
    return r.json()["message"]["content"]


# How many CHunks formed in following document (handbook)
chunks = []

for file in Path("docs").glob("*.md"):
    chunks += [
        p.strip()
        for p in file.read_text(encoding="utf-8").split("\n\n")
        if p.strip()
    ]
    
    chunk_vectors = embed(chunks)
    print(f"Indexed{len(chunks)} chunks\n")

    # how many chinks we want to return for any given question

def retrieve(question, k=3):
    q = embed([question])[0]
    scores = chunk_vectors @ q / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]

# calling LLms

while True:
    question = input("Ask: ")
    context = "\n\n".join(retrieve(question))
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}"""
    print("\n" + ask_llm(prompt) + "\n")