# Project Prompt — Multimodal RAG System (paste this into a new chat)

I'm building a **Multimodal RAG (Retrieval-Augmented Generation) application** as a resume project.
I'm a CS/AI-ML undergrad with experience in Python, PyTorch, FastAPI, Streamlit, Docker, OpenCV,
and a prior single-modal RAG project (text-only study assistant). I'm working alone, on a Mac,
in VS Code. I want this project to be strong enough to showcase to recruiters for
AI/ML Engineer roles.

## Goal
Build a RAG system that answers questions using a PDF/slide deck that contains **text, tables,
AND images/charts** — not just plain text chunks like a typical RAG tutorial project. The system
should retrieve the most relevant content (text or image) and generate a grounded answer, citing
which page/element it came from.

## Functional requirements
1. **Ingestion pipeline**: parse PDFs, split into text chunks, extract tables, and extract
   embedded images/charts as separate elements (page number + bounding box metadata).
2. **Multimodal embeddings**: embed text chunks with a text embedding model, and embed images
   with a vision-capable embedding model (e.g., CLIP) so both live in a comparable vector space
   (or use two separate indexes with a fusion/re-ranking step — you decide and justify).
3. **Retrieval**: given a user question, retrieve top-k relevant text chunks AND top-k relevant
   images/tables, then combine into context.
4. **Generation**: pass combined multimodal context to an LLM (local via Ollama/MLX or an API)
   to produce a grounded answer, including a citation like "page 4, Figure 2."
5. **Answer grounding / guardrails**: if no relevant content is found, say so rather than
   hallucinating.
6. **API layer**: FastAPI backend exposing `/ingest` and `/ask` endpoints.
7. **Frontend**: simple Streamlit (or React, your call) UI to upload a PDF and chat with it,
   showing the retrieved image/table alongside the answer.
8. **Evaluation**: a small eval script/notebook that measures retrieval quality (e.g., manual
   relevance labels on 15–20 questions) and answer faithfulness.

## Non-functional requirements
- Must run reasonably well on a Mac (Apple Silicon). Prefer **MLX** or **Ollama** for any local
  model inference over CUDA-only tooling. If using a hosted LLM API instead, make that swappable
  via a config/env var, and keep local-only fallback options documented.
- Clean, modular code (`backend/app/{api,core,services,models}` structure already scaffolded —
  use it).
- Dockerized for deployment.
- A clear, well-illustrated README with an architecture diagram (Mermaid is fine), setup
  instructions, and example queries/screenshots.
- Deploy a live demo (Hugging Face Spaces, Render, or Fly.io) if feasible with the chosen models.

## What I want from you (the assistant) in this chat
1. Start by proposing a concrete tech stack (embedding models, vector store, parsing library,
   LLM choice) optimized for running on a Mac, with brief tradeoffs — don't just assume; ask me
   if something is genuinely a toss-up.
2. Then build this step by step, starting with the ingestion pipeline, verifying each stage
   works (e.g., show me extracted chunks/images) before moving to retrieval, then generation,
   then the API, then the frontend, then Docker, then eval.
3. Write real, runnable code into files (not just snippets) inside the folder structure I've set
   up (I'll upload it / it's at `/home/claude/multimodal-rag-project` if this is a computer-use
   session, otherwise recreate the structure).
4. Explain key design decisions briefly as you go, since I want to be able to talk about this
   project confidently in interviews — I should understand *why* each component was chosen, not
   just have working code.
5. At the end, help me write the README (with architecture diagram) and a short "what I'd do
   next" section, since interviewers often ask about extensions/limitations.

## Constraints
- No CUDA-only dependencies unless there's a CPU/MPS fallback.
- Keep API costs low — prefer free/local models where quality allows, and clearly flag any step
  that requires a paid API key.
- I'll test everything in VS Code on my Mac as we go, so give me terminal commands for each step.

Let's start with step 1: propose the tech stack.
