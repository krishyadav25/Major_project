"""FastAPI application entry point for Adapted Reasoning RAG."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.routes import router as core_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events."""
    # --- Startup ---
    configure_logging()
    logger.info(
        "Starting %s (v%s) in '%s' environment",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    yield
    # --- Shutdown ---
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Application factory for the FastAPI backend."""
    app = FastAPI(
        title="Adapted Reasoning RAG API",
        description=(
            "Bilingual, voice-enabled welfare scheme eligibility assistant "
            "with claim-level evidence sufficiency checking."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Cross-Origin Resource Sharing (CORS) for frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(core_router)

    return app


# Application instance for ASGI servers (e.g., uvicorn app.main:app)
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
    )
