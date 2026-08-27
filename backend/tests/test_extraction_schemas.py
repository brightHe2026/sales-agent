import pytest
from pydantic import ValidationError

from app.enums.memory import OwnerType
from app.schemas.memory import ProjectSignal, StructuredActivityExtraction, TaskCandidate


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_schema_confidence_boundaries(value):
    result = StructuredActivityExtraction(summary="ok", overall_confidence=value)
    assert result.overall_confidence == value


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_schema_rejects_invalid_confidence(value):
    with pytest.raises(ValidationError):
        StructuredActivityExtraction(summary="bad", overall_confidence=value)


def test_unknown_owner_is_not_fabricated():
    candidate = TaskCandidate(title="Confirm environment", confidence=0.5)
    assert candidate.owner_type is OwnerType.UNKNOWN
    assert candidate.owner_name is None


@pytest.mark.parametrize("empty_text", ["", "   ", "\t\n"])
def test_empty_candidate_text_and_empty_project_signal_are_rejected(empty_text):
    with pytest.raises(ValidationError):
        TaskCandidate(title=empty_text, confidence=0.5)
    with pytest.raises(ValidationError):
        StructuredActivityExtraction(summary=empty_text, overall_confidence=0.5)
    with pytest.raises(ValidationError):
        ProjectSignal(project_name=empty_text, confidence=0.5)
