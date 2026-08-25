from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.enums.memory import OwnerType, Priority, RiskSeverity


class CandidateBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ProjectSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str | None = None
    customer_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class RequirementCandidate(CandidateBase):
    requirement_type: str | None = None
    priority: Priority | None = None


class TaskCandidate(CandidateBase):
    owner_type: OwnerType = OwnerType.UNKNOWN
    owner_name: str | None = None
    due_at: datetime | None = None
    priority: Priority | None = None


class DecisionCandidate(CandidateBase):
    decision_maker: str | None = None
    decided_at: datetime | None = None


class RiskCandidate(CandidateBase):
    severity: RiskSeverity = RiskSeverity.UNKNOWN
    mitigation: str | None = None


class StructuredActivityExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    project_signal: ProjectSignal | None = None
    requirements: list[RequirementCandidate] = Field(default_factory=list)
    tasks: list[TaskCandidate] = Field(default_factory=list)
    decisions: list[DecisionCandidate] = Field(default_factory=list)
    risks: list[RiskCandidate] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool = False
