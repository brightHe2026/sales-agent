import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.enums.memory import ActivityType, ExtractionStatus, SourceType


class Participant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    role: str | None = Field(default=None, max_length=255)


class ActivityCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    activity_type: ActivityType
    occurred_at: datetime
    raw_content: str = Field(min_length=1)
    summary: str | None = None
    source_type: SourceType
    source_ref: str | None = None
    participants: list[Participant] | None = None

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value

    @field_validator("raw_content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_content must not be blank")
        return value


class ActivityRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    customer_id: uuid.UUID | None
    project_id: uuid.UUID | None
    activity_type: ActivityType
    occurred_at: datetime
    raw_content: str
    summary: str | None
    source_type: SourceType
    source_ref: str | None
    participants: list[Participant] | None
    extraction_status: ExtractionStatus
    extraction_version: str | None
    extraction_confidence: float | None
    created_at: datetime
    updated_at: datetime
