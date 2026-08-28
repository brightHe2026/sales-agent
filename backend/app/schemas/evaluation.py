from datetime import datetime
from typing import Literal

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
    fact_aliases: dict[str, list[str]] = Field(default_factory=dict)

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

    @model_validator(mode="after")
    def validate_fact_aliases(self) -> "EvaluationCase":
        signal = self.expected.project_signal
        canonical: dict[str, set[str]] = {
            "project": {signal.project_name} if signal and signal.project_name else set(),
            "customer": {signal.customer_name} if signal and signal.customer_name else set(),
            "requirement": {item.title for item in self.expected.requirements},
            "task": {item.title for item in self.expected.tasks},
            "decision": {item.title for item in self.expected.decisions},
            "risk": {item.title for item in self.expected.risks},
        }
        valid_keys = {
            f"{kind}:{' '.join(title.casefold().split())}"
            for kind, titles in canonical.items()
            for title in titles
        }
        unknown_keys = set(self.fact_aliases) - valid_keys
        if unknown_keys:
            raise ValueError(f"fact_aliases contains unknown labels: {sorted(unknown_keys)}")
        accepted: dict[str, str] = {key: key for key in valid_keys}
        for key, values in self.fact_aliases.items():
            kind = key.split(":", 1)[0]
            for value in values:
                normalized = " ".join(value.casefold().split())
                if not normalized:
                    raise ValueError("fact aliases must not be blank")
                alias_key = f"{kind}:{normalized}"
                existing = accepted.get(alias_key)
                if existing is not None and existing != key:
                    raise ValueError(f"fact alias {value!r} maps to multiple facts")
                accepted[alias_key] = key
        return self


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


FactKind = Literal["project", "customer", "requirement", "task", "decision", "risk"]


class EquivalentFactMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    kind: FactKind
    expected_title: str = Field(min_length=1)
    actual_title: str = Field(min_length=1)


class EvaluationAdjudication(BaseModel):
    """Human-approved semantic equivalences bound to immutable input artifacts."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matches: list[EquivalentFactMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_one_to_one_matches(self) -> "EvaluationAdjudication":
        expected = [
            (item.case_id, item.kind, " ".join(item.expected_title.casefold().split()))
            for item in self.matches
        ]
        actual = [
            (item.case_id, item.kind, " ".join(item.actual_title.casefold().split()))
            for item in self.matches
        ]
        if len(expected) != len(set(expected)):
            raise ValueError("an expected fact may be adjudicated only once")
        if len(actual) != len(set(actual)):
            raise ValueError("an actual fact may be adjudicated only once")
        return self


class PostHocMetrics(BaseModel):
    """Adjudicated metrics intentionally omit the independent Gate verdict."""

    dataset_name: str
    case_count: int
    precision: float
    recall: float
    f1: float
    owner_accuracy: float
    review_required_accuracy: float
    hallucinated_facts: int
    cases: list[EvaluationCaseResult]


class PostHocAdjudicationReport(BaseModel):
    report_type: Literal["post_hoc_adjudication"] = "post_hoc_adjudication"
    independent_holdout: Literal[False] = False
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjudication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strict_report: EvaluationReport
    adjudicated_metrics: PostHocMetrics
