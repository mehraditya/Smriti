from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_qdrant
from api.routers.memory import router as memory_router
from config import get_settings
from storage.postgres_client import check_db_health, init_db
from storage.qdrant_client import QdrantStore

log = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: init DB tables + Qdrant collection. Shutdown: close connections."""
    log.info("app.startup", env=settings.debug and "dev" or "prod")

    await init_db()

    qdrant = get_qdrant()
    await qdrant.ensure_collection()

    log.info("app.ready")
    yield

    await qdrant.close()
    log.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Memory System API",
        description="Stateful graph-based memory backend",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(memory_router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(qdrant: Annotated[QdrantStore, Depends(get_qdrant)]) -> HealthResponse:
        pg_ok = await check_db_health()
        qdrant_ok = await qdrant.check_health()
        status = "ok" if (pg_ok and qdrant_ok) else "degraded"
        return HealthResponse(status=status, postgres=pg_ok, qdrant=qdrant_ok)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"service": "memory-system", "version": "0.1.0"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )