from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from app.database import get_session
from app.main import app
from app.schemas.memory.activity import ActivityCreate


def test_activity_ingestion_api(session):
    def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session
    try:
        client = TestClient(app)
        response = client.post(
            "/activities",
            json={
                "activity_type": "MANUAL_NOTE",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "raw_content": "Original customer note",
                "source_type": "MANUAL",
            },
        )
        get_response = client.get(f"/activities/{response.json()['id']}")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 201
    assert response.json()["raw_content"] == "Original customer note"
    assert response.json()["extraction_status"] == "PENDING"
    assert get_response.status_code == 200
    assert get_response.json()["id"] == response.json()["id"]


def test_activity_api_rejects_blank_raw_content(session):
    def override_session():
        yield session

    app.dependency_overrides[get_session] = override_session
    try:
        response = TestClient(app).post(
            "/activities",
            json={
                "activity_type": "MANUAL_NOTE",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "raw_content": "   ",
                "source_type": "MANUAL",
            },
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422


def test_activity_input_rejects_naive_datetime_and_extra_fields():
    with pytest.raises(ValidationError):
        ActivityCreate(
            activity_type="MANUAL_NOTE",
            occurred_at="2026-08-26T09:00:00",
            raw_content="note",
            source_type="MANUAL",
        )
    with pytest.raises(ValidationError):
        ActivityCreate(
            activity_type="MANUAL_NOTE",
            occurred_at="2026-08-26T09:00:00+08:00",
            raw_content="note",
            source_type="MANUAL",
            unexpected="not allowed",
        )
