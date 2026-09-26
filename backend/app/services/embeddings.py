"""
Embedding models for the multimodal RAG pipeline.

Text:
    BAAI/bge-small-en-v1.5 -> 384-dimensional embeddings

Images:
    google/siglip-base-patch16-224 -> 768-dimensional embeddings
"""

from pathlib import Path
from functools import lru_cache

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor


TEXT_MODEL = "BAAI/bge-small-en-v1.5"
IMAGE_MODEL = "google/siglip-base-patch16-224"


def _get_device() -> str:
    """Use Apple Silicon GPU through MPS when available."""
    if torch.backends.mps.is_available():
        return "mps"

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_text_embedder() -> SentenceTransformer:
    """Load the BGE text embedding model once."""
    return SentenceTransformer(
        TEXT_MODEL,
        device=_get_device(),
    )


class SigLIPEmbedder:
    """Small wrapper around SigLIP for image embeddings."""

    def __init__(self) -> None:
        self.device = _get_device()
        self.processor = AutoProcessor.from_pretrained(IMAGE_MODEL)
        self.model = AutoModel.from_pretrained(IMAGE_MODEL)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_image(self, image_path: str | Path) -> list[float]:
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model.get_image_features(**inputs)
        
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs
        
        # Normalize for cosine-similarity retrieval.
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        
        return features[0].detach().cpu().tolist()
    @torch.inference_mode()
    def encode_text(self, text: str) -> list[float]:
        """Convert text into a SigLIP embedding."""

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model.get_text_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=-1,
        )

        return features[0].detach().cpu().tolist()   
@lru_cache(maxsize=1)
def get_image_embedder() -> SigLIPEmbedder:
    """Load SigLIP once and reuse it."""
    return SigLIPEmbedder()