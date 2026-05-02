import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import Document, DocumentChunk


async def list_documents(team_id: uuid.UUID, db: AsyncSession) -> list[Document]:
    result = await db.execute(
        select(Document)
        .where(Document.team_id == team_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def delete_document(doc_id: uuid.UUID, team_id: uuid.UUID, db: AsyncSession) -> None:
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.team_id == team_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise ValueError("Document not found")
    await db.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id))
    await db.delete(document)
    await db.commit()
