import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Enum, Float, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.enums.memory import (
    ActivityType,
    ExtractionStatus,
    OwnerType,
    Priority,
    ProjectStage,
    ProjectStatus,
    RequirementStatus,
    RiskSeverity,
    RiskStatus,
    SourceType,
    TaskStatus,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class UUIDMixin:
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)


class Customer(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "customers"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)

    projects: Mapped[list["Project"]] = relationship(back_populates="customer")
    activities: Mapped[list["Activity"]] = relationship(back_populates="customer")


class Project(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "projects"

    customer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("customers.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    stage: Mapped[ProjectStage] = mapped_column(Enum(ProjectStage, name="project_stage"), nullable=False)
    status: Mapped[ProjectStatus] = mapped_column(Enum(ProjectStatus, name="project_status"), nullable=False)
    expected_timeline: Mapped[str | None] = mapped_column(Text)
    next_action: Mapped[str | None] = mapped_column(Text)

    customer: Mapped[Customer] = relationship(back_populates="projects")
    activities: Mapped[list["Activity"]] = relationship(back_populates="project", passive_deletes=True)
    requirements: Mapped[list["Requirement"]] = relationship(back_populates="project")
    tasks: Mapped[list["Task"]] = relationship(back_populates="project")
    decisions: Mapped[list["Decision"]] = relationship(back_populates="project")
    risks: Mapped[list["Risk"]] = relationship(back_populates="project")


class Activity(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "activities"
    __table_args__ = (
        CheckConstraint(
            "extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)",
            name="ck_activities_extraction_confidence",
        ),
    )

    customer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL")
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL")
    )
    activity_type: Mapped[ActivityType] = mapped_column(Enum(ActivityType, name="activity_type"), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    source_type: Mapped[SourceType] = mapped_column(Enum(SourceType, name="source_type"), nullable=False)
    source_ref: Mapped[str | None] = mapped_column(Text)
    participants: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    extraction_status: Mapped[ExtractionStatus] = mapped_column(
        Enum(ExtractionStatus, name="extraction_status"), default=ExtractionStatus.PENDING, nullable=False
    )
    extraction_version: Mapped[str | None] = mapped_column(String(100))
    extraction_confidence: Mapped[float | None] = mapped_column(Float)

    customer: Mapped[Customer | None] = relationship(back_populates="activities")
    project: Mapped[Project | None] = relationship(back_populates="activities")


class DerivedFactMixin(UUIDMixin, TimestampMixin):
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), nullable=False)
    source_activity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("activities.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    source_fingerprint: Mapped[str | None] = mapped_column(String(64))


class Requirement(DerivedFactMixin, Base):
    __tablename__ = "requirements"
    __table_args__ = (
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_requirements_confidence"),
        UniqueConstraint("source_activity_id", "source_fingerprint", name="uq_requirements_source_fingerprint"),
    )

    description: Mapped[str] = mapped_column(Text, nullable=False)
    requirement_type: Mapped[str | None] = mapped_column(String(100))
    priority: Mapped[Priority | None] = mapped_column(Enum(Priority, name="priority"))
    status: Mapped[RequirementStatus] = mapped_column(Enum(RequirementStatus, name="requirement_status"), nullable=False)
    project: Mapped[Project] = relationship(back_populates="requirements")


class Task(DerivedFactMixin, Base):
    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_tasks_confidence"),
        UniqueConstraint("source_activity_id", "source_fingerprint", name="uq_tasks_source_fingerprint"),
    )

    description: Mapped[str | None] = mapped_column(Text)
    owner_type: Mapped[OwnerType] = mapped_column(Enum(OwnerType, name="owner_type"), nullable=False)
    owner_name: Mapped[str | None] = mapped_column(String(255))
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    priority: Mapped[Priority | None] = mapped_column(Enum(Priority, name="priority"))
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus, name="task_status"), nullable=False)
    project: Mapped[Project] = relationship(back_populates="tasks")


class Decision(DerivedFactMixin, Base):
    __tablename__ = "decisions"
    __table_args__ = (
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_decisions_confidence"),
        UniqueConstraint("source_activity_id", "source_fingerprint", name="uq_decisions_source_fingerprint"),
    )

    description: Mapped[str] = mapped_column(Text, nullable=False)
    decision_maker: Mapped[str | None] = mapped_column(String(255))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    project: Mapped[Project] = relationship(back_populates="decisions")


class Risk(DerivedFactMixin, Base):
    __tablename__ = "risks"
    __table_args__ = (
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_risks_confidence"),
        UniqueConstraint("source_activity_id", "source_fingerprint", name="uq_risks_source_fingerprint"),
    )

    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[RiskSeverity] = mapped_column(Enum(RiskSeverity, name="risk_severity"), nullable=False)
    status: Mapped[RiskStatus] = mapped_column(Enum(RiskStatus, name="risk_status"), nullable=False)
    mitigation: Mapped[str | None] = mapped_column(Text)
    project: Mapped[Project] = relationship(back_populates="risks")
