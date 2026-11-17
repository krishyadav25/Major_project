"""Integration tests for application startup, lifespan, and API docs."""

from app.main import create_app
from fastapi.testclient import TestClient


def test_app_startup_and_lifespan() -> None:
    """Test that the application starts up and shuts down cleanly within lifespan context."""
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_openapi_schema_available() -> None:
    """Test that the OpenAPI JSON specification is generated and accessible."""
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert data["info"]["title"] == "Adapted Reasoning RAG API"
        assert "/health" in data["paths"]


def test_swagger_ui_available() -> None:
    """Test that the Swagger documentation UI is accessible at /docs."""
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.get("/docs")
        assert response.status_code == 200
        assert "swagger-ui" in response.text.lower()


def test_cors_headers_present() -> None:
    """Test that CORS headers are configured for cross-origin requests."""
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.status_code == 200
        # Starlette echoes back the origin when allow_credentials=True
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
