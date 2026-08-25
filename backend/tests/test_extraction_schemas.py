import pytest
from pydantic import ValidationError

from app.enums.memory import OwnerType
from app.schemas.memory import StructuredActivityExtraction, TaskCandidate


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
