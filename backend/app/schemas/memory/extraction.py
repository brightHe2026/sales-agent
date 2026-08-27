from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.enums.memory import OwnerType, Priority, RiskSeverity


NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CandidateBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: NonBlankText
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ProjectSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: NonBlankText | None = None
    customer_name: NonBlankText | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_signal(self) -> "ProjectSignal":
        if self.project_name is None and self.customer_name is None:
            raise ValueError("project_signal requires a project or customer name")
        return self


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

    summary: NonBlankText
    project_signal: ProjectSignal | None = None
    requirements: list[RequirementCandidate] = Field(default_factory=list)
    tasks: list[TaskCandidate] = Field(default_factory=list)
    decisions: list[DecisionCandidate] = Field(default_factory=list)
    risks: list[RiskCandidate] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool = False
