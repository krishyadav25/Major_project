"""Unit tests for the health check endpoint and core configuration."""

from app.core.config import get_settings
from app.core.routes import HealthResponse
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check_status_code() -> None:
    """Test that GET /health returns HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_response_body() -> None:
    """Test that GET /health returns the exact expected response fields and values."""
    response = client.get("/health")
    data = response.json()

    assert data == {
        "status": "healthy",
        "service": "adapted-reasoning-rag",
    }
    assert data["status"] == "healthy"
    assert data["service"] == "adapted-reasoning-rag"


def test_health_check_schema_validation() -> None:
    """Test that the response matches the HealthResponse Pydantic model."""
    response = client.get("/health")
    validated = HealthResponse(**response.json())

    assert validated.status == "healthy"
    assert validated.service == "adapted-reasoning-rag"


def test_settings_defaults() -> None:
    """Test that default configuration values are properly set."""
    settings = get_settings()

    assert settings.app_name == "adapted-reasoning-rag"
    assert settings.app_version == "0.1.0"
    assert settings.app_env in {"development", "testing", "production"}
    assert settings.backend_port == 8000
    assert settings.backend_host == "0.0.0.0"
    assert settings.log_level.upper() == "INFO"
