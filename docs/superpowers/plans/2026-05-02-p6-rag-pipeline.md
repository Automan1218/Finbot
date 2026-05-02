# P6 RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the high-quality RAG pipeline (embedding cache → ingestion → hybrid search + RRF → query rewrite + HyDE → reranker → agent tool) on top of the existing `documents` / `document_chunks` schema.

**Architecture:** All RAG logic lives under `app/rag/` with one file per responsibility. Embeddings and rewrite queries cache in Redis (Layer 4 + Layer 5 from spec §6.2). Hybrid search runs dense (pgvector HNSW cosine) and sparse (PostgreSQL `tsv` GIN) in parallel and fuses via RRF; reranker calls `gpt-4o-mini` for cross-encoder scoring. A new `smart_retrieve` router branches on `is_vague` between HyDE and Rewrite+Hybrid. Knowledge base API exposes upload/list/delete; agent gains a `rag_retrieve` tool.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async, asyncpg, pgvector, Redis (`redis.asyncio`), OpenAI Python SDK (`AsyncOpenAI`).

---

## File Structure

**New files (under `backend/app/`):**
- `rag/__init__.py` — package marker
- `rag/embedding.py` — `get_embedding(text)` with Redis cache (Layer 4)
- `rag/ingest.py` — chunk text + persist `Document` + `DocumentChunk` rows
- `rag/hybrid_search.py` — `hybrid_search` (dense + sparse + RRF)
- `rag/rewrite.py` — `rewrite_query` with Redis cache (Layer 5)
- `rag/hyde.py` — `hyde_retrieve`
- `rag/reranker.py` — `rerank` cross-encoder via gpt-4o-mini
- `rag/retrieve.py` — `smart_retrieve` router
- `rag/schemas.py` — Pydantic schemas for document upload/list responses
- `rag/router.py` — `/knowledge-base/documents` endpoints
- `rag/service.py` — `create_document`, `list_documents`, `delete_document`

**New tests (under `backend/tests/`):**
- `test_rag_embedding.py`
- `test_rag_ingest.py`
- `test_rag_hybrid_search.py`
- `test_rag_rewrite.py`
- `test_rag_hyde.py`
- `test_rag_reranker.py`
- `test_rag_retrieve.py`
- `test_knowledge_router.py`

**Modified files:**
- `backend/app/main.py` — register `rag_router`
- `backend/app/agent/tools.py` — add `RAG_RETRIEVE_TOOL` schema + intent
- `backend/app/agent/executor.py` — handle `rag_retrieve` intent
- `backend/app/agent/llm.py` — `normalize_intent` accepts `rag_retrieve`
- `backend/app/core/config.py` — add `OPENAI_EMBEDDING_MODEL`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`

---

### Task 1: Embedding helper with Redis cache

**Files:**
- Create: `backend/app/rag/__init__.py`
- Create: `backend/app/rag/embedding.py`
- Create: `backend/tests/test_rag_embedding.py`
- Modify: `backend/app/core/config.py`

- [ ] **Step 1: Add embedding model setting**

Edit `backend/app/core/config.py`, add field after `OPENAI_MODEL`:

```python
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
```

- [ ] **Step 2: Create rag package**

Create `backend/app/rag/__init__.py` with empty content.

- [ ] **Step 3: Write failing test for cache miss path**

Create `backend/tests/test_rag_embedding.py`:

```python
import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.embedding import _cache_key, get_embedding


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self.store[key] = value


def _floats_to_bytes(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _bytes_to_floats(payload: bytes) -> list[float]:
    count = len(payload) // 4
    return list(struct.unpack(f"{count}f", payload))


@pytest.mark.asyncio
async def test_cache_miss_calls_openai_and_stores_bytes():
    redis = FakeRedis()
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    vector = await get_embedding("hello", redis=redis, client=fake_client)

    assert vector == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    fake_client.embeddings.create.assert_awaited_once()
    cached = await redis.get(_cache_key("hello"))
    assert _bytes_to_floats(cached) == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_cache_hit_skips_openai():
    redis = FakeRedis()
    redis.store[_cache_key("hi")] = _floats_to_bytes([0.5, 0.6])
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock()

    vector = await get_embedding("hi", redis=redis, client=fake_client)

    assert vector == [pytest.approx(0.5), pytest.approx(0.6)]
    fake_client.embeddings.create.assert_not_awaited()
```

- [ ] **Step 4: Run test to verify it fails**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_embedding.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.rag.embedding'`

- [ ] **Step 5: Implement embedding helper**

Create `backend/app/rag/embedding.py`:

```python
import hashlib
import struct
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.redis import get_redis


def _cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"embed:{digest}"


def _floats_to_bytes(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _bytes_to_floats(payload: bytes) -> list[float]:
    count = len(payload) // 4
    return list(struct.unpack(f"{count}f", payload))


async def get_embedding(
    text: str,
    redis: Any = None,
    client: AsyncOpenAI | None = None,
) -> list[float]:
    redis = redis if redis is not None else await get_redis()
    key = _cache_key(text)
    cached = await redis.get(key)
    if cached:
        return _bytes_to_floats(cached)

    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    vector = list(response.data[0].embedding)
    await redis.set(key, _floats_to_bytes(vector))
    return vector
```

- [ ] **Step 6: Run test to verify pass**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_embedding.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```
git add backend/app/core/config.py backend/app/rag/__init__.py backend/app/rag/embedding.py backend/tests/test_rag_embedding.py
git commit -m "feat: P6 RAG embedding helper with Redis cache"
```

---

### Task 2: Document chunking + ingestion service

**Files:**
- Create: `backend/app/rag/ingest.py`
- Create: `backend/tests/test_rag_ingest.py`

- [ ] **Step 1: Write failing test for chunking**

Create `backend/tests/test_rag_ingest.py`:

```python
import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.models.knowledge import Document, DocumentChunk
from app.rag.ingest import chunk_text, ingest_document


def test_chunk_text_splits_with_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) >= 3
    assert chunks[0] == text[:30]
    assert chunks[1].startswith(text[25:30])  # 5-char overlap
    joined = "".join(c[5:] if i > 0 else c for i, c in enumerate(chunks))
    assert text in joined or joined.startswith(text[:30])


def test_chunk_text_returns_single_chunk_when_short():
    chunks = chunk_text("short", chunk_size=100, overlap=10)
    assert chunks == ["short"]


@pytest.mark.asyncio
async def test_ingest_document_persists_doc_and_chunks(db_session, finance_setup, monkeypatch):
    user, team, _ = finance_setup
    fake_embed = AsyncMock(side_effect=lambda text, **_: [0.1] * 1536)
    monkeypatch.setattr("app.rag.ingest.get_embedding", fake_embed)

    doc = await ingest_document(
        team_id=team.id,
        user_id=user.id,
        title="Policy",
        source_type="policy",
        text="abcdefghij" * 100,
        db=db_session,
        chunk_size=200,
        overlap=20,
    )

    assert isinstance(doc, Document)
    chunks = (
        await db_session.execute(
            select(DocumentChunk).where(DocumentChunk.doc_id == doc.id).order_by(DocumentChunk.chunk_index)
        )
    ).scalars().all()
    assert len(chunks) >= 5
    assert all(len(c.embedding) == 1536 for c in chunks)
    assert chunks[0].chunk_index == 0


@pytest.mark.asyncio
async def test_ingest_document_rejects_empty_text(db_session, finance_setup):
    user, team, _ = finance_setup
    with pytest.raises(ValueError):
        await ingest_document(
            team_id=team.id,
            user_id=user.id,
            title="Empty",
            source_type="policy",
            text="   ",
            db=db_session,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_ingest.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement chunking + ingestion**

Create `backend/app/rag/ingest.py`:

```python
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.knowledge import Document, DocumentChunk
from app.rag.embedding import get_embedding


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

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
```

- [ ] **Step 4: Run test to verify pass**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_ingest.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/rag/ingest.py backend/tests/test_rag_ingest.py
git commit -m "feat: P6 RAG document chunking + ingestion service"
```

---

### Task 3: Knowledge base router + service

**Files:**
- Create: `backend/app/rag/schemas.py`
- Create: `backend/app/rag/service.py`
- Create: `backend/app/rag/router.py`
- Create: `backend/tests/test_knowledge_router.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing test for router**

Create `backend/tests/test_knowledge_router.py`:

```python
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_create_document_returns_201(finance_setup, monkeypatch):
    user, team, token = finance_setup
    fake_embed = AsyncMock(return_value=[0.1] * 1536)
    monkeypatch.setattr("app.rag.ingest.get_embedding", fake_embed)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            f"/knowledge-base/documents?team_id={team.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "team_id": str(team.id),
                "title": "Reimbursement Policy",
                "source_type": "policy",
                "text": "出差报销限额：机票经济舱，酒店每晚500元以内。" * 10,
            },
        )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Reimbursement Policy"
    assert data["source_type"] == "policy"


@pytest.mark.asyncio
async def test_list_documents_returns_team_docs(finance_setup, monkeypatch):
    user, team, token = finance_setup
    fake_embed = AsyncMock(return_value=[0.1] * 1536)
    monkeypatch.setattr("app.rag.ingest.get_embedding", fake_embed)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.post(
            f"/knowledge-base/documents?team_id={team.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "team_id": str(team.id),
                "title": "Doc A",
                "source_type": "manual",
                "text": "policy content " * 50,
            },
        )
        response = await ac.get(
            f"/knowledge-base/documents?team_id={team.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    docs = response.json()
    assert any(d["title"] == "Doc A" for d in docs)


@pytest.mark.asyncio
async def test_create_document_requires_admin(finance_setup, db_session, monkeypatch):
    from app.core.security import create_access_token, hash_password
    from app.models.team import TeamMember
    from app.models.user import User
    import uuid

    _, team, _ = finance_setup
    member = User(email=f"viewer_{uuid.uuid4().hex[:6]}@test.com", password_hash=hash_password("pw"))
    db_session.add(member)
    await db_session.flush()
    db_session.add(TeamMember(team_id=team.id, user_id=member.id, role="viewer"))
    await db_session.commit()
    viewer_token = create_access_token(str(member.id))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            f"/knowledge-base/documents?team_id={team.id}",
            headers={"Authorization": f"Bearer {viewer_token}"},
            json={
                "team_id": str(team.id),
                "title": "Should fail",
                "source_type": "policy",
                "text": "x" * 100,
            },
        )
    assert response.status_code == 403
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_knowledge_router.py -v
```
Expected: FAIL with import errors

- [ ] **Step 3: Create schemas**

Create `backend/app/rag/schemas.py`:

```python
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    team_id: uuid.UUID
    title: str = Field(min_length=1, max_length=200)
    source_type: str = Field(min_length=1, max_length=50)
    text: str = Field(min_length=1)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    team_id: uuid.UUID
    title: str
    source_type: str
    created_by: uuid.UUID
    created_at: datetime
```

- [ ] **Step 4: Create service**

Create `backend/app/rag/service.py`:

```python
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
    doc = result.scalar_one_or_none()
    if not doc:
        raise ValueError("Document not found")
    await db.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id))
    await db.delete(doc)
    await db.commit()
```

- [ ] **Step 5: Create router**

Create `backend/app/rag/router.py`:

```python
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.rag import schemas, service
from app.rag.ingest import ingest_document
from app.teams.service import get_member_role

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])

_ADMIN_ROLES = ("owner", "admin")
_ANY_ROLES = ("owner", "admin", "member", "viewer")


async def _require_team(
    team_id: uuid.UUID = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[uuid.UUID, User]:
    role = await get_member_role(team_id, current_user.id, db)
    if role not in _ANY_ROLES:
        raise HTTPException(status_code=403, detail="Team not found or access denied")
    return team_id, current_user


async def _require_admin(
    team_id: uuid.UUID = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[uuid.UUID, User]:
    role = await get_member_role(team_id, current_user.id, db)
    if role not in _ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Admin or owner required")
    return team_id, current_user


@router.post("/documents", response_model=schemas.DocumentResponse, status_code=201)
async def create_document(
    body: schemas.DocumentCreate,
    ctx: tuple = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
):
    team_id, user = ctx
    if body.team_id != team_id:
        raise HTTPException(status_code=400, detail="team_id mismatch")
    return await ingest_document(
        team_id=team_id,
        user_id=user.id,
        title=body.title,
        source_type=body.source_type,
        text=body.text,
        db=db,
    )


@router.get("/documents", response_model=list[schemas.DocumentResponse])
async def list_docs(
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    return await service.list_documents(team_id, db)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_doc(
    doc_id: uuid.UUID,
    ctx: tuple = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    try:
        await service.delete_document(doc_id, team_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Step 6: Register router**

Edit `backend/app/main.py`. Add import after existing routers and include statement:

```python
from app.rag.router import router as rag_router
```

And after `app.include_router(teams_router)`:

```python
app.include_router(rag_router)
```

- [ ] **Step 7: Run tests to verify pass**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_knowledge_router.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```
git add backend/app/rag/schemas.py backend/app/rag/service.py backend/app/rag/router.py backend/app/main.py backend/tests/test_knowledge_router.py
git commit -m "feat: P6 knowledge base router + service"
```

---

### Task 4: Hybrid Search (Dense + Sparse + RRF)

**Files:**
- Create: `backend/app/rag/hybrid_search.py`
- Create: `backend/tests/test_rag_hybrid_search.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_rag_hybrid_search.py`:

```python
from unittest.mock import AsyncMock

import pytest

from app.rag.hybrid_search import hybrid_search, reciprocal_rank_fusion
from app.rag.ingest import ingest_document


def test_rrf_ranks_overlapping_doc_higher():
    dense = [{"id": "a", "chunk_text": "x"}, {"id": "b", "chunk_text": "y"}]
    sparse = [{"id": "b", "chunk_text": "y"}, {"id": "c", "chunk_text": "z"}]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert fused[0]["id"] == "b"
    assert {d["id"] for d in fused} == {"a", "b", "c"}


def test_rrf_handles_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []


@pytest.mark.asyncio
async def test_hybrid_search_returns_matching_chunks(db_session, finance_setup, monkeypatch):
    user, team, _ = finance_setup
    fake_embed = AsyncMock(side_effect=lambda text, **_: [0.1] * 1536)
    monkeypatch.setattr("app.rag.ingest.get_embedding", fake_embed)
    monkeypatch.setattr("app.rag.hybrid_search.get_embedding", fake_embed)

    await ingest_document(
        team_id=team.id,
        user_id=user.id,
        title="Travel Policy",
        source_type="policy",
        text="出差报销机票经济舱酒店500元以内餐饮150元",
        db=db_session,
        chunk_size=200,
        overlap=10,
    )

    results = await hybrid_search("出差", team.id, db_session, top_k=5)
    assert len(results) >= 1
    assert any("出差" in r["chunk_text"] for r in results)
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_hybrid_search.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement hybrid search**

Create `backend/app/rag/hybrid_search.py`:

```python
import asyncio
import uuid
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embedding import get_embedding


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for rank, doc in enumerate(dense):
        scores[str(doc["id"])] = scores.get(str(doc["id"]), 0.0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(sparse):
        scores[str(doc["id"])] = scores.get(str(doc["id"]), 0.0) + 1.0 / (k + rank + 1)

    ids_sorted = sorted(scores, key=lambda x: scores[x], reverse=True)
    doc_map: dict[str, dict[str, Any]] = {}
    for doc in dense + sparse:
        doc_map.setdefault(str(doc["id"]), doc)
    return [doc_map[i] for i in ids_sorted if i in doc_map]


async def _dense_search(
    embedding: list[float],
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int,
) -> list[dict[str, Any]]:
    sql = text(
        """
        SELECT dc.id, dc.chunk_text,
               1 - (dc.embedding <=> CAST(:embedding AS vector)) AS score
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.doc_id
        WHERE d.team_id = :team_id AND dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
        """
    )
    embedding_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
    result = await db.execute(
        sql,
        {"embedding": embedding_literal, "team_id": str(team_id), "top_k": top_k},
    )
    return [
        {"id": row.id, "chunk_text": row.chunk_text, "score": float(row.score)}
        for row in result
    ]


async def _sparse_search(
    query: str,
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int,
) -> list[dict[str, Any]]:
    sql = text(
        """
        SELECT dc.id, dc.chunk_text,
               ts_rank(dc.tsv, plainto_tsquery('simple', :query)) AS score
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.doc_id
        WHERE d.team_id = :team_id
          AND dc.tsv @@ plainto_tsquery('simple', :query)
        ORDER BY score DESC
        LIMIT :top_k
        """
    )
    result = await db.execute(
        sql,
        {"query": query, "team_id": str(team_id), "top_k": top_k},
    )
    return [
        {"id": row.id, "chunk_text": row.chunk_text, "score": float(row.score)}
        for row in result
    ]


async def hybrid_search(
    query: str,
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    embedding = await get_embedding(query)
    dense_task = _dense_search(embedding, team_id, db, top_k)
    sparse_task = _sparse_search(query, team_id, db, top_k)
    dense, sparse = await asyncio.gather(dense_task, sparse_task)
    return reciprocal_rank_fusion(dense, sparse)
```

- [ ] **Step 4: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_hybrid_search.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/rag/hybrid_search.py backend/tests/test_rag_hybrid_search.py
git commit -m "feat: P6 hybrid search dense+sparse+RRF"
```

---

### Task 5: Query Rewrite with cache

**Files:**
- Create: `backend/app/rag/rewrite.py`
- Create: `backend/tests/test_rag_rewrite.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_rag_rewrite.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.rewrite import _cache_key, rewrite_query


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value.encode("utf-8")


def _build_client(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_rewrite_caches_result():
    redis = FakeRedis()
    client = _build_client("差旅费报销 流程 审批 金额阈值")

    rewritten = await rewrite_query("这笔要审批吗", redis=redis, client=client)
    assert "审批" in rewritten

    cached = await rewrite_query("这笔要审批吗", redis=redis, client=client)
    assert cached == rewritten
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_rewrite_cache_hit_skips_llm():
    redis = FakeRedis()
    redis.store[_cache_key("hi")] = b"already-rewritten"
    client = _build_client("should-not-be-used")

    result = await rewrite_query("hi", redis=redis, client=client)
    assert result == "already-rewritten"
    client.chat.completions.create.assert_not_awaited()
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_rewrite.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement rewrite**

Create `backend/app/rag/rewrite.py`:

```python
import hashlib
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.redis import get_redis

REWRITE_PROMPT = """将用户问题改写为更适合文档检索的形式。
扩展缩写，补全背景，保留核心意图。只输出改写后的问题，不要解释。

原问题: {query}
改写:"""

_TTL_SECONDS = 3600


def _cache_key(query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"rewrite:{digest}"


async def rewrite_query(
    query: str,
    redis: Any = None,
    client: AsyncOpenAI | None = None,
) -> str:
    redis = redis if redis is not None else await get_redis()
    key = _cache_key(query)
    cached = await redis.get(key)
    if cached:
        return cached.decode("utf-8") if isinstance(cached, (bytes, bytearray)) else str(cached)

    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
    )
    rewritten = (response.choices[0].message.content or query).strip()
    await redis.setex(key, _TTL_SECONDS, rewritten)
    return rewritten
```

- [ ] **Step 4: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_rewrite.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/rag/rewrite.py backend/tests/test_rag_rewrite.py
git commit -m "feat: P6 query rewrite with Redis cache"
```

---

### Task 6: HyDE retrieval

**Files:**
- Create: `backend/app/rag/hyde.py`
- Create: `backend/tests/test_rag_hyde.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_rag_hyde.py`:

```python
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.rag.hyde import hyde_retrieve


def _build_client(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_hyde_calls_llm_then_dense_search(monkeypatch):
    client = _build_client("出差报销限额：机票经济舱，酒店每晚500元以内。")
    fake_embed = AsyncMock(return_value=[0.1] * 1536)
    fake_dense = AsyncMock(return_value=[{"id": "x", "chunk_text": "policy"}])
    monkeypatch.setattr("app.rag.hyde.get_embedding", fake_embed)
    monkeypatch.setattr("app.rag.hyde._dense_search", fake_dense)

    fake_db = MagicMock()
    team_id = uuid.uuid4()
    results = await hyde_retrieve("出差有什么限制", team_id, fake_db, client=client)

    assert results == [{"id": "x", "chunk_text": "policy"}]
    client.chat.completions.create.assert_awaited_once()
    fake_embed.assert_awaited_once()
    fake_dense.assert_awaited_once()
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_hyde.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement HyDE**

Create `backend/app/rag/hyde.py`:

```python
import uuid
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.rag.embedding import get_embedding
from app.rag.hybrid_search import _dense_search

HYDE_PROMPT = """假设你是财务政策文档，写一段回答以下问题的内容（50字以内，直接陈述事实，不要解释）：
{query}"""


async def hyde_retrieve(
    query: str,
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int = 20,
    client: AsyncOpenAI | None = None,
) -> list[dict[str, Any]]:
    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
    )
    hypothesis = (response.choices[0].message.content or query).strip()
    embedding = await get_embedding(hypothesis)
    return await _dense_search(embedding, team_id, db, top_k)
```

- [ ] **Step 4: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_hyde.py -v
```
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```
git add backend/app/rag/hyde.py backend/tests/test_rag_hyde.py
git commit -m "feat: P6 HyDE retrieval"
```

---

### Task 7: Reranker

**Files:**
- Create: `backend/app/rag/reranker.py`
- Create: `backend/tests/test_rag_reranker.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_rag_reranker.py`:

```python
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.reranker import rerank


def _build_client(scores: list[float]) -> MagicMock:
    payload = json.dumps({"scores": scores})
    msg = MagicMock(content=payload)
    choice = MagicMock(message=msg)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_rerank_returns_top_n_in_order():
    candidates = [
        {"id": "a", "chunk_text": "alpha"},
        {"id": "b", "chunk_text": "beta"},
        {"id": "c", "chunk_text": "gamma"},
    ]
    client = _build_client([1.0, 9.5, 4.0])

    ranked = await rerank("query", candidates, top_n=2, client=client)

    assert [d["id"] for d in ranked] == ["b", "c"]


@pytest.mark.asyncio
async def test_rerank_empty_returns_empty():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    result = await rerank("q", [], top_n=5, client=client)
    assert result == []
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_falls_back_when_response_invalid():
    candidates = [{"id": "a", "chunk_text": "alpha"}, {"id": "b", "chunk_text": "beta"}]
    msg = MagicMock(content="not-json")
    choice = MagicMock(message=msg)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    result = await rerank("q", candidates, top_n=2, client=client)
    assert result == candidates[:2]
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_reranker.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement reranker**

Create `backend/app/rag/reranker.py`:

```python
import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings


_PROMPT_TEMPLATE = (
    "对以下文档片段按与问题的相关性打分（0-10）。"
    "只输出 JSON 对象，格式 {{\"scores\": [..]}}，长度={count}。\n"
    "问题：{query}\n"
    "文档：{docs}"
)


async def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    client: AsyncOpenAI | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    docs_text = json.dumps(
        [c["chunk_text"] for c in candidates],
        ensure_ascii=False,
    )
    prompt = _PROMPT_TEMPLATE.format(
        count=len(candidates), query=query, docs=docs_text
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        scores = json.loads(content).get("scores") or []
        if len(scores) != len(candidates):
            raise ValueError("score length mismatch")
    except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
        return candidates[:top_n]

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]
```

- [ ] **Step 4: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_reranker.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/rag/reranker.py backend/tests/test_rag_reranker.py
git commit -m "feat: P6 cross-encoder reranker"
```

---

### Task 8: smart_retrieve router

**Files:**
- Create: `backend/app/rag/retrieve.py`
- Create: `backend/tests/test_rag_retrieve.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_rag_retrieve.py`:

```python
from unittest.mock import AsyncMock
import uuid

import pytest

from app.rag.retrieve import is_vague_query, smart_retrieve


def test_is_vague_short_query():
    assert is_vague_query("出差") is True


def test_is_vague_question_words():
    assert is_vague_query("出差报销有什么限制") is True


def test_is_vague_specific_query():
    assert is_vague_query("差旅费报销标准2026年最新版本说明") is False


@pytest.mark.asyncio
async def test_smart_retrieve_uses_hyde_for_vague(monkeypatch):
    fake_hyde = AsyncMock(return_value=[{"id": "h", "chunk_text": "hyde"}])
    fake_hybrid = AsyncMock(return_value=[{"id": "x", "chunk_text": "hybrid"}])
    fake_rewrite = AsyncMock(return_value="should-not-be-called")
    fake_rerank = AsyncMock(side_effect=lambda q, c, top_n, **_: c[:top_n])
    monkeypatch.setattr("app.rag.retrieve.hyde_retrieve", fake_hyde)
    monkeypatch.setattr("app.rag.retrieve.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.rag.retrieve.rewrite_query", fake_rewrite)
    monkeypatch.setattr("app.rag.retrieve.rerank", fake_rerank)

    results = await smart_retrieve("出差有啥限制", uuid.uuid4(), db=None, top_n=3)

    assert results == [{"id": "h", "chunk_text": "hyde"}]
    fake_hyde.assert_awaited_once()
    fake_hybrid.assert_not_awaited()
    fake_rewrite.assert_not_awaited()


@pytest.mark.asyncio
async def test_smart_retrieve_uses_rewrite_hybrid_for_specific(monkeypatch):
    fake_hyde = AsyncMock()
    fake_hybrid = AsyncMock(return_value=[{"id": "x", "chunk_text": "hybrid"}])
    fake_rewrite = AsyncMock(return_value="rewritten")
    fake_rerank = AsyncMock(side_effect=lambda q, c, top_n, **_: c[:top_n])
    monkeypatch.setattr("app.rag.retrieve.hyde_retrieve", fake_hyde)
    monkeypatch.setattr("app.rag.retrieve.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.rag.retrieve.rewrite_query", fake_rewrite)
    monkeypatch.setattr("app.rag.retrieve.rerank", fake_rerank)

    results = await smart_retrieve(
        "差旅费报销标准2026最新版", uuid.uuid4(), db=None, top_n=5
    )

    assert results == [{"id": "x", "chunk_text": "hybrid"}]
    fake_rewrite.assert_awaited_once()
    assert fake_rewrite.await_args.args[0] == "差旅费报销标准2026最新版"
    fake_hybrid.assert_awaited_once()
    assert fake_hybrid.await_args.args[0] == "rewritten"
    fake_hyde.assert_not_awaited()
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_retrieve.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement smart_retrieve**

Create `backend/app/rag/retrieve.py`:

```python
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.hybrid_search import hybrid_search
from app.rag.hyde import hyde_retrieve
from app.rag.reranker import rerank
from app.rag.rewrite import rewrite_query


_VAGUE_WORDS = ("怎么", "什么", "有没有", "能不能", "如何", "为什么", "啥")


def is_vague_query(query: str) -> bool:
    text = query.strip()
    if len(text) < 15:
        return True
    return any(word in text for word in _VAGUE_WORDS)


async def smart_retrieve(
    query: str,
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int = 20,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if is_vague_query(query):
        candidates = await hyde_retrieve(query, team_id, db, top_k=top_k)
    else:
        rewritten = await rewrite_query(query)
        candidates = await hybrid_search(rewritten, team_id, db, top_k=top_k)

    if not candidates:
        return []
    return await rerank(query, candidates, top_n=top_n)
```

- [ ] **Step 4: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_rag_retrieve.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/rag/retrieve.py backend/tests/test_rag_retrieve.py
git commit -m "feat: P6 smart_retrieve query router"
```

---

### Task 9: rag_retrieve agent tool integration

**Files:**
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/app/agent/llm.py`
- Modify: `backend/app/agent/executor.py`
- Modify: `backend/tests/test_agent_executor.py`

- [ ] **Step 1: Add rag_retrieve tool schema**

Edit `backend/app/agent/tools.py`. Update the `AgentIntent` Literal and add tool definition. Replace existing class:

```python
class AgentIntent(TypedDict):
    name: Literal["record_transaction", "generate_report", "rag_retrieve", "clarify"]
    arguments: dict[str, Any]
```

Add new tool below `CLARIFY_TOOL`:

```python
RAG_RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_retrieve",
        "description": "Look up team policy / knowledge documents to answer a question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}
```

Update `FINBOT_TOOLS`:

```python
FINBOT_TOOLS = [RECORD_TRANSACTION_TOOL, GENERATE_REPORT_TOOL, RAG_RETRIEVE_TOOL, CLARIFY_TOOL]
```

- [ ] **Step 2: Update normalize_intent**

Edit `backend/app/agent/llm.py`. In `normalize_intent`, add a branch before the final fallback:

```python
    if name == "rag_retrieve":
        return {
            "name": "rag_retrieve",
            "arguments": {"query": str(arguments.get("query") or original_message)},
        }
```

- [ ] **Step 3: Handle rag_retrieve in executor**

Edit `backend/app/agent/executor.py`. Add import at top:

```python
from app.rag.retrieve import smart_retrieve
```

Update `execute_intent` to add a branch:

```python
async def execute_intent(
    intent: AgentIntent,
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    if intent["name"] == "record_transaction":
        return await _record_transaction(intent["arguments"], team_id, user_id, db)
    if intent["name"] == "generate_report":
        return await _generate_report(intent["arguments"], team_id, user_id, db)
    if intent["name"] == "rag_retrieve":
        return await _rag_retrieve(intent["arguments"], team_id, db)
    return {
        "status": "needs_clarification",
        "message": intent["arguments"]["question"],
        "missing_fields": intent["arguments"].get("missing_fields", []),
    }
```

Add the new helper at the bottom of the file:

```python
async def _rag_retrieve(
    args: dict[str, Any],
    team_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        return {
            "status": "needs_clarification",
            "message": "Please describe what you want to look up.",
            "missing_fields": ["query"],
        }
    chunks = await smart_retrieve(query, team_id, db)
    return {
        "status": "retrieved",
        "message": f"Retrieved {len(chunks)} relevant chunks.",
        "chunks": [
            {"id": str(c.get("id")), "chunk_text": c.get("chunk_text")} for c in chunks
        ],
    }
```

- [ ] **Step 4: Add executor test**

Edit `backend/tests/test_agent_executor.py`. Append:

```python
@pytest.mark.asyncio
async def test_execute_rag_retrieve_returns_chunks(db_session, setup, monkeypatch):
    user, team = setup
    fake_retrieve = AsyncMock(
        return_value=[{"id": "doc-1", "chunk_text": "policy snippet"}]
    )
    monkeypatch.setattr("app.agent.executor.smart_retrieve", fake_retrieve)

    intent = {"name": "rag_retrieve", "arguments": {"query": "出差报销"}}
    result = await execute_intent(intent, team.id, user.id, db_session)

    assert result["status"] == "retrieved"
    assert result["chunks"][0]["chunk_text"] == "policy snippet"
    fake_retrieve.assert_awaited_once()
```

Make sure `from unittest.mock import AsyncMock` exists at the top of the file (add it if not present).

- [ ] **Step 5: Run agent tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_executor.py backend/tests/test_agent_llm.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```
git add backend/app/agent/tools.py backend/app/agent/llm.py backend/app/agent/executor.py backend/tests/test_agent_executor.py
git commit -m "feat: P6 wire rag_retrieve into agent executor"
```

---

### Task 10: End-to-end full suite verification

**Files:** none new — verification only.

- [ ] **Step 1: Run full test suite**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/ -v
```
Expected: all green; new tests count >= old + 16.

- [ ] **Step 2: If any test fails, fix root cause and re-run**

Check for: import cycles between `rag/hybrid_search.py` and `rag/hyde.py`, missing `bindparam` usage causing pgvector cast errors, fixture ordering issues. Address only the failing test.

- [ ] **Step 3: Final commit**

If any fix needed:

```
git add -A
git commit -m "fix: P6 follow-up adjustments after full-suite run"
```

If suite passes cleanly first time, no commit needed — milestone done.

---

## Self-Review

**Spec coverage:**
- §5.1 Pipeline → Task 4 (hybrid+RRF), Task 5 (rewrite), Task 6 (hyde), Task 7 (rerank), Task 8 (smart_retrieve)
- §5.2 HNSW/IVFFlat indexes → already in P1 migration; no work needed
- §5.3 Hybrid Search → Task 4
- §5.4 Query Rewrite → Task 5 (with cache per §6.2 Layer 5)
- §5.5 HyDE → Task 6
- §5.6 Query Analysis routing → Task 8 (`is_vague_query`)
- §5.7 Reranker → Task 7
- §6.2 Layer 4 Embedding cache → Task 1
- §11 Knowledge base API (POST/GET/DELETE /knowledge-base/documents) → Task 3
- §12 Permission rules (admin+ for upload) → Task 3 (`_require_admin`)
- §4.2 `rag_retrieve` agent tool → Task 9

**Out of scope (handled in P7+ milestones):**
- §6.1/§6.2 Layer 1-3 (LLM response cache, budget cache, category cache) — P7 Redis cache layer
- §6.3 Session window compression — P7
- §7 TTFT optimization (parallel context, prefix caching, streaming) — P8
- §8 Eval / RAGAS / A/B — P9
- `record_batch` / `analyze_spending` agent tools — P10 frontend / agent expansion
- `transactions.embedding` IVFFlat semantic search — P10 (column exists; no consumer in P6)

**Placeholder scan:** none.

**Type consistency:**
- `AgentIntent.name` Literal extended in Task 9; matches `execute_intent` branches; matches `normalize_intent` returns.
- `hybrid_search`, `hyde_retrieve`, `smart_retrieve` all return `list[dict[str, Any]]` with `id` + `chunk_text` keys.
- `_dense_search` reused by both `hybrid_search` and `hyde_retrieve` (one source of truth for dense SQL).
- Embedding cached as float32 bytes, decoded back via `struct.unpack` — symmetric.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-02-p6-rag-pipeline.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review.
2. **Inline Execution** — same session, batched checkpoints.

Which approach?
