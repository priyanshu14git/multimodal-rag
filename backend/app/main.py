from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="Multimodal RAG API",
    description="Multimodal PDF Retrieval-Augmented Generation API",
    version="1.0.0",
)


app.include_router(router)
