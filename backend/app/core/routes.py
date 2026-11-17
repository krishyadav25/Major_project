"""Core system and diagnostic routes."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Schema for the health check endpoint response."""

    status: str = Field(
        default="healthy",
        description="Current operational status of the service",
        examples=["healthy"],
    )
    service: str = Field(
        default="adapted-reasoning-rag",
        description="Name of the service",
        examples=["adapted-reasoning-rag"],
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the operational status and service identifier.",
)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
    )
