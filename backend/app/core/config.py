"""
Centralized configuration.

Everything that could plausibly change between your laptop, a teammate's
machine, or a deployed instance lives here and is overridable via
environment variables (see `.env.example` at the project root). This is the
one file you edit to swap models without touching pipeline code.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    # --- Paths -------------------------------------------------------
    raw_pdfs_dir: Path = DATA_DIR / "raw_pdfs"
    processed_dir: Path = DATA_DIR / "processed"
    vectorstore_dir: Path = DATA_DIR / "vectorstore"

    # --- Chunking ------------------------------------------------------
    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 150

    # --- Embedding models ----------------------------------------------
    # Text embedder: small, fast, strong MTEB retrieval score, CPU/MPS friendly.
    text_embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Image embedder: SigLIP outperforms CLIP at a comparable compute budget
    # and has solid MPS support via transformers on Apple Silicon.
    image_embedding_model: str = "google/siglip-base-patch16-224"
    embedding_device: str = "mps"  # falls back to "cpu" automatically if MPS unavailable

    # --- Vector store ----------------------------------------------------
    text_collection_name: str = "text_chunks"
    image_collection_name: str = "image_elements"

    # --- Retrieval ---------------------------------------------------------
    top_k_text: int = 4
    top_k_images: int = 2
    # Similarity floor below which we consider "nothing relevant found"
    # (used for the answer-grounding guardrail). Tune this once you run eval.
    min_relevance_score: float = 0.25

    # --- Generation (LLM) ------------------------------------------------
    # Kept swappable even though the default path is local-only Ollama, so
    # switching to a hosted API later is a one-line env change, not a
    # code change.
    llm_provider: str = "ollama"  # ollama | anthropic | openai
    ollama_model: str = "llava:7b"
    ollama_host: str = "http://localhost:11434"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    def ensure_data_dirs(self) -> None:
        for d in (self.raw_pdfs_dir, self.processed_dir, self.vectorstore_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
