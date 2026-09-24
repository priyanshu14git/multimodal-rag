# Multimodal RAG Project — Starter Scaffold

This is a starter folder to open in VS Code. It contains:

- `PROJECT_PROMPT.md` — **paste this whole file into a new Claude chat** to kick off the build.
  It tells Claude exactly what to build, how, and in what order.
- `backend/app/{api,core,services,models}` — empty modular structure for the FastAPI backend
  (ingestion, retrieval, generation logic will go here).
- `backend/data/{raw_pdfs,processed,vectorstore}` — where your source PDFs, processed
  chunks/images, and vector index will live (kept out of git via `.gitignore`).
- `frontend/` — for the Streamlit or React UI.
- `notebooks/` — for experimentation and the retrieval/answer evaluation notebook.
- `tests/` — unit tests for ingestion/retrieval/generation.
- `docs/` — architecture diagram, screenshots, write-up for your README/portfolio.

## How to use this
1. Unzip this folder and open it in VS Code (`code multimodal-rag-project`).
2. Open a **new Claude chat** and paste in the contents of `PROJECT_PROMPT.md`.
3. If that new chat has computer-use / file tools, tell it the scaffold already exists at this
   path (or re-upload the zip) so it builds directly into these folders instead of starting from
   scratch.
4. Work through it step by step — ingestion → retrieval → generation → API → frontend → Docker → eval.
5. Commit early and often (`git init` now, before you write code, so your commit history shows
   real progress — recruiters do check this).

## Suggested first terminal commands
```bash
cd multimodal-rag-project
git init
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```
