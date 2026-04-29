from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.auth.router import router as auth_router
from app.finance.router import router as finance_router
from app.teams.router import router as teams_router
from app.core.database import AsyncSessionLocal, engine
from app.core.redis import close_redis, get_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()
    await close_redis()


app = FastAPI(title="Finbot API", version="0.1.0", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(finance_router)
app.include_router(teams_router)


@app.get("/health")
async def health():
    db_ok = False
    redis_ok = False

    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    try:
        r = await get_redis()
        await r.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if db_ok and redis_ok else "degraded",
        "db": "ok" if db_ok else "error",
        "redis": "ok" if redis_ok else "error",
    }
