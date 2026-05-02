import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.knowledge import Document, DocumentChunk
from app.rag.embedding import get_embedding


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


async def ingest_document(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    title: str,
    source_type: str,
    text: str,
    db: AsyncSession,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> Document:
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.RAG_CHUNK_OVERLAP
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Document text is empty after stripping")

    document = Document(
        team_id=team_id,
        title=title,
        source_type=source_type,
        created_by=user_id,
    )
    db.add(document)
    await db.flush()

    for index, piece in enumerate(chunks):
        embedding = await get_embedding(piece)
        db.add(
            DocumentChunk(
                doc_id=document.id,
                chunk_text=piece,
                embedding=embedding,
                chunk_index=index,
            )
        )

    await db.commit()
    await db.refresh(document)
    return document
