import asyncio
import logging 

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException, RequestValidationError

from src.config.database import connect_db, close_db, get_db
from src.config.settings import get_settings
from src.middleware.error_middleware import (
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from src.middleware.request_context import RequestContextMiddleware
from src.middleware.logging_middleware import LoggingMiddleware
from src.middleware.security_headers import SecurityHeadersMiddleware
from src.routes.agent import router as agent_router
from src.routes.ml import router as ml_router
from src.routes.dashboard import api_router
from src.services.ml.artifact_loader import load_artifacts
from src.services.rag.vector_store import collection_count
from src.middleware.request_guard import RequestGuardMiddleware


ENV_SETTINGS = get_settings()

logging.basicConfig(
    level=logging.DEBUG if ENV_SETTINGS.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s (debug=%s)",
        ENV_SETTINGS.app_name,
        ENV_SETTINGS.app_version,
        ENV_SETTINGS.debug,
    )

    await connect_db()
    load_artifacts()

    indexed_vectors = collection_count()
    if indexed_vectors == 0:
        logger.warning(
            "Index RAG kosong — endpoint /chat akan menjawab NOT_FOUND. "
            "Jalankan `python -m jobs.reindex_rag` untuk mengisinya.")
    else:
        logger.info("Index RAG siap: %d vektor", indexed_vectors)

    yield

    await close_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title=ENV_SETTINGS.app_name,
    version=ENV_SETTINGS.app_version,
    lifespan=lifespan,
    docs_url="/docs" if ENV_SETTINGS.environment != "production" else None,
    redoc_url="/redoc" if ENV_SETTINGS.environment != "production" else None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ENV_SETTINGS.cors_origins_raw.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RequestGuardMiddleware,
    limits={
        "/api/v1/chat": (ENV_SETTINGS.chat_rate_limit,
                         ENV_SETTINGS.rate_limit_period),
        "/api/v1": (ENV_SETTINGS.rate_limit, ENV_SETTINGS.rate_limit_period),
    },
    max_body_bytes=ENV_SETTINGS.max_request_bytes,
)

app.add_middleware(LoggingMiddleware)

app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    SecurityHeadersMiddleware,
    hsts=(ENV_SETTINGS.environment == "production"),
)



app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(ml_router)
app.include_router(agent_router)
app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "environment": ENV_SETTINGS.environment}




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=ENV_SETTINGS.api_host,
        port=ENV_SETTINGS.api_port,
        reload=ENV_SETTINGS.debug,
    )
