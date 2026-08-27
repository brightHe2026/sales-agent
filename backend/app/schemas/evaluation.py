from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.enums.memory import ActivityType, SourceType
from app.schemas.memory.activity import Participant
from app.schemas.memory.extraction import StructuredActivityExtraction


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    raw_content: str = Field(min_length=1)
    activity_type: ActivityType
    source_type: SourceType = SourceType.MANUAL
    occurred_at: datetime
    participants: list[Participant] | None = None
    expected: StructuredActivityExtraction

    @field_validator("raw_content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_content must not be blank")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_cases_and_expected_facts(self) -> "EvaluationDataset":
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case ids must be unique")
        if not any(
            case.expected.project_signal
            or
            case.expected.requirements
            or case.expected.tasks
            or case.expected.decisions
            or case.expected.risks
            for case in self.cases
        ):
            raise ValueError("evaluation dataset must contain at least one expected fact")
        for case in self.cases:
            for kind in ("requirements", "tasks", "decisions", "risks"):
                titles = [
                    " ".join(item.title.casefold().split())
                    for item in getattr(case.expected, kind)
                ]
                if len(titles) != len(set(titles)):
                    raise ValueError(
                        f"case {case.id} has duplicate normalized {kind} titles"
                    )
        return self


class EvaluationThresholds(BaseModel):
    min_precision: float = Field(default=0.9, ge=0.0, le=1.0)
    min_recall: float = Field(default=0.85, ge=0.0, le=1.0)
    min_owner_accuracy: float = Field(default=0.9, ge=0.0, le=1.0)
    min_review_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    max_hallucinated_facts: int = Field(default=0, ge=0)


class EvaluationCaseResult(BaseModel):
    case_id: str
    expected_facts: int
    actual_facts: int
    matched_facts: int
    hallucinated_facts: int
    owner_checks: int
    owner_matches: int
    review_required_match: bool
    missing_facts: list[str]
    hallucinated_fact_labels: list[str]


class EvaluationReport(BaseModel):
    dataset_name: str
    case_count: int
    precision: float
    recall: float
    f1: float
    owner_accuracy: float
    review_required_accuracy: float
    hallucinated_facts: int
    passed: bool
    cases: list[EvaluationCaseResult]
