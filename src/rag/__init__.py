"""A small, local Retrieval-Augmented Generation (RAG) application."""

from pathlib import Path

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"
REQUEST_TIMEOUT = (5, 120)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"


def embed(texts: list[str]) -> np.ndarray:
    """Convert one or more text strings into embedding vectors."""
    if not texts:
        raise ValueError("Cannot create embeddings from an empty list.")

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    embeddings = response.json()["embeddings"]
    if len(embeddings) != len(texts):
        raise ValueError("Ollama returned a different number of embeddings than requested.")

    return np.asarray(embeddings, dtype=np.float32)


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[str]:
    """Load Markdown files and split them into paragraph-sized chunks."""
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {docs_dir}. Create it and add .md files."
        )

    chunks: list[str] = []
    for file_path in sorted(docs_dir.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        chunks.extend(
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        )

    if not chunks:
        raise ValueError(f"No Markdown content found in {docs_dir}.")

    return chunks


def retrieve(
    question: str,
    chunks: list[str],
    chunk_vectors: np.ndarray,
    k: int = 3,
) -> list[str]:
    """Return the k chunks with the highest cosine similarity to a question."""
    if not question.strip():
        return []
    if k < 1:
        raise ValueError("k must be at least 1.")

    question_vector = embed([question])[0]
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
    question_norm = np.linalg.norm(question_vector)

    if question_norm == 0:
        return []

    denominator = chunk_norms * question_norm
    scores = np.divide(
        chunk_vectors @ question_vector,
        denominator,
        out=np.zeros_like(chunk_norms),
        where=denominator != 0,
    )

    best_indices = np.argsort(scores)[::-1][:k]
    return [chunks[index] for index in best_indices]


def ask_llm(prompt: str) -> str:
    """Send a prompt to the local Ollama chat model."""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def main() -> None:
    """Load documents, build the in-memory index, and start the chat loop."""
    print("Loading documents...")
    try:
        chunks = load_chunks()
        print(f"Loaded {len(chunks)} chunks.")

        print("Creating embeddings with Ollama...")
        # Embed once after all documents have been loaded.
        chunk_vectors = embed(chunks)
        print("Index ready. Ask a question (type 'exit' to quit).\n")

        while True:
            try:
                question = input("Ask: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if not question:
                continue

            relevant_chunks = retrieve(question, chunks, chunk_vectors)
            if not relevant_chunks:
                print("\nI don't know.\n")
                continue

            context = "\n\n".join(relevant_chunks)
            prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".
Do not make up facts.

Context:
{context}

Question: {question}"""

            print("\n" + ask_llm(prompt) + "\n")

    except requests.RequestException as error:
        print(
            "\nCould not connect to Ollama or complete the model request. "
            "Make sure Ollama is running and both models are available."
        )
        print(f"Details: {error}")
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"\nSetup error: {error}")


if __name__ == "__main__":
    main()
