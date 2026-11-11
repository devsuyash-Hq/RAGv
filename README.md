# RAGv – Local Retrieval-Augmented Generation (RAG) with Ollama

A small RAG app that runs **fully on your own computer**. It answers questions from your own Markdown documents, with no API key and no cloud service.

## Demo

```
Ask: How many casual leave days do employees receive per year?
Employees receive 12 casual leave days per year.
```

## How it works

```
docs/*.md → split into paragraphs → embeddings (nomic-embed-text)
                                          │
Question → embedding → cosine similarity → top 3 chunks
                                          │
        qwen3:4b-instruct (question + chunks) → answer
```

1. Loads all `.md` files from the `docs/` folder
2. Splits them into paragraph-sized chunks
3. Creates embeddings with Ollama (`nomic-embed-text`) and keeps them in memory
4. For each question, finds the 3 most similar chunks using cosine similarity (NumPy)
5. Sends the question and chunks to `qwen3:4b-instruct`, which must answer **only from the context** or say "I don't know"

## Tech stack

- Python 3
- [Ollama](https://ollama.com) (local LLM and embeddings)
- NumPy, Requests
- [uv](https://github.com/astral-sh/uv) for dependency management

## Project structure

```
RAGv/
├── docs/             # your Markdown documents
├── src/rag/          # main RAG code
├── Rag.py            # backward-compatible entry point
├── pyproject.toml
└── README.md
```

## Getting started

**1. Install Ollama** from https://ollama.com and pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
```

**2. Clone and install**

```bash
git clone https://github.com/devsuyash-Hq/RAGv.git
cd RAGv
uv sync
```

**3. Add your documents**

Put one or more `.md` files in the `docs/` folder.

**4. Run**

```bash
uv run rag
```

Type `exit` to quit.

## Example questions

| Question | Answer |
|---|---|
| How many casual leave days do employees receive per year? | 12 days |

## Features

- Runs 100% locally and privately
- Answers only from your documents and says "I don't know" otherwise
- Clear errors if Ollama isn't running or the `docs/` folder is missing

## What I learned

- How the retrieve-then-generate flow works
- How embeddings and cosine similarity find relevant text
- How to ground an LLM's answers in documents to reduce made-up answers

## Future improvements

- [ ] Show the source chunk for each answer
- [ ] Overlapping chunks for better retrieval
- [ ] PDF support
- [ ] Streamlit UI

## Author

**Suyash** – Java developer and MCA student
GitHub: [@devsuyash-Hq](https://github.com/devsuyash-Hq)
