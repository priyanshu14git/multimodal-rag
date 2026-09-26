from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import AskRequest
from app.services import vectorstore
from app.services.embeddings import (
    get_image_embedder,
    get_text_embedder,
)
from app.services.pdf_parser import parse_pdf
from app.services.rag import answer_multimodal_question


router = APIRouter()


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PDF_DIR = PROJECT_ROOT / "backend" / "data" / "raw_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "multimodal-rag-api",
    }


@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = RAW_PDF_DIR / Path(file.filename).name

    try:
        contents = await file.read()
        pdf_path.write_bytes(contents)

        manifest = parse_pdf(
            pdf_path=pdf_path,
            processed_dir=PROCESSED_DIR,
        )

        client = vectorstore.get_client()
        text_embedder = get_text_embedder()
        image_embedder = get_image_embedder()

        counts = vectorstore.index_manifest(
            manifest=manifest,
            client=client,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
        )

        return {
            "message": "PDF ingested successfully.",
            "doc_id": manifest.doc_id,
            "source_filename": manifest.source_filename,
            "num_pages": manifest.num_pages,
            "text_chunks": counts["text_chunks"],
            "tables": counts["tables"],
            "images": counts["images"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.post("/ask")
def ask_question(request: AskRequest):
    try:
        result = answer_multimodal_question(
            question=request.question,
            doc_id=request.doc_id,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
