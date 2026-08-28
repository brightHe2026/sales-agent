import asyncio
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.enums.memory import ActivityType, ExtractionStatus, OwnerType, SourceType
from app.extraction import PydanticActivityExtractor, PydanticEvidenceGrounder
from app.extraction.evidence import EvidenceGroundingError
from app.extraction.pydantic_ai import EXTRACTION_INSTRUCTIONS
from app.models import Activity, Decision, Requirement, Risk, Task
from app.repositories.memory import ActivityRepository
from app.schemas.memory.extraction import StructuredActivityExtraction, TaskCandidate
from app.services.activity_extraction import (
    ActivityExtractionError,
    ActivityExtractionService,
    ActivityForExtractionNotFoundError,
)


def make_activity(session, raw_content="I will provide the POC plan by Friday."):
    activity = Activity(
        activity_type=ActivityType.MANUAL_NOTE,
        occurred_at=datetime.now(timezone.utc),
        raw_content=raw_content,
        source_type=SourceType.MANUAL,
        extraction_status=ExtractionStatus.PENDING,
    )
    return ActivityRepository(session).create(activity)


class StubExtractor:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def extract(self, activity):
        if self.error:
            raise self.error
        return self.result


def test_service_persists_extraction_metadata_but_not_candidate_facts(session):
    activity = make_activity(session)
    result = StructuredActivityExtraction(
        summary="POC plan promised for Friday.",
        tasks=[TaskCandidate(title="Provide POC plan", owner_type=OwnerType.SELF, confidence=0.9)],
        overall_confidence=0.9,
    )
    service = ActivityExtractionService(
        ActivityRepository(session), StubExtractor(result=result), extraction_version="task003-v1"
    )
    returned = service.extract(activity.id)
    session.refresh(activity)
    assert returned == result
    assert activity.raw_content == "I will provide the POC plan by Friday."
    assert activity.summary == result.summary
    assert activity.extraction_status is ExtractionStatus.PROCESSED
    assert activity.extraction_version == "task003-v1"
    assert activity.extraction_confidence == 0.9
    assert session.query(Requirement).count() == 0
    assert session.query(Task).count() == 0
    assert session.query(Decision).count() == 0
    assert session.query(Risk).count() == 0


def test_review_required_result_sets_review_status(session):
    activity = make_activity(session, "Someone should follow up soon.")
    result = StructuredActivityExtraction(
        summary="Follow-up mentioned without a clear owner or date.",
        tasks=[TaskCandidate(title="Follow up", confidence=0.4)],
        overall_confidence=0.4,
        review_required=True,
    )
    service = ActivityExtractionService(
        ActivityRepository(session), StubExtractor(result=result), extraction_version="task003-v1"
    )
    service.extract(activity.id)
    assert activity.extraction_status is ExtractionStatus.REVIEW_REQUIRED
    assert result.tasks[0].owner_type is OwnerType.UNKNOWN


def test_extraction_failure_is_recorded_and_wrapped(session):
    activity = make_activity(session)
    service = ActivityExtractionService(
        ActivityRepository(session), StubExtractor(error=TimeoutError("model timeout")), extraction_version="task003-v1"
    )
    with pytest.raises(ActivityExtractionError) as exc_info:
        service.extract(activity.id)
    session.refresh(activity)
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert activity.extraction_status is ExtractionStatus.FAILED
    assert activity.extraction_version == "task003-v1"
    assert activity.raw_content == "I will provide the POC plan by Friday."


def test_missing_activity_is_rejected(session):
    service = ActivityExtractionService(
        ActivityRepository(session), StubExtractor(), extraction_version="task003-v1"
    )
    with pytest.raises(ActivityForExtractionNotFoundError):
        service.extract("00000000-0000-0000-0000-000000000001")


def test_pydantic_ai_enforces_typed_output_without_network(session):
    activity = make_activity(session)

    def model_function(messages, info: AgentInfo):
        assert "I will provide the POC plan by Friday." in str(messages)
        output_tool = info.output_tools[0]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tool.name,
                    {
                        "summary": "POC plan promised for Friday.",
                        "project_signal": None,
                        "requirements": [],
                        "tasks": [
                            {
                                "title": "Provide POC plan",
                                "description": None,
                                "confidence": 0.95,
                                "owner_type": "SELF",
                                "owner_name": None,
                                "due_at": None,
                                "priority": None,
                            }
                        ],
                        "decisions": [],
                        "risks": [],
                        "overall_confidence": 0.95,
                        "review_required": False,
                    },
                )
            ]
        )

    extractor = PydanticActivityExtractor(FunctionModel(model_function))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = extractor.extract(activity)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    assert isinstance(result, StructuredActivityExtraction)
    assert result.tasks[0].owner_type is OwnerType.SELF
    assert result.overall_confidence == 0.95


def test_extraction_prompt_version_and_grounding_rules_are_explicit():
    assert PydanticActivityExtractor.prompt_version == "structured-memory-v1"
    assert "Extract only facts explicitly supported" in EXTRACTION_INSTRUCTIONS
    assert "do not turn completed work into a task" in EXTRACTION_INSTRUCTIONS
    assert "SELF only when" in EXTRACTION_INSTRUCTIONS


def test_evidence_grounder_only_attaches_exact_quotes(session):
    activity = make_activity(session)
    extraction = StructuredActivityExtraction(
        summary="POC plan promised for Friday.",
        tasks=[TaskCandidate(title="Provide POC plan", owner_type=OwnerType.SELF, confidence=0.9)],
        overall_confidence=0.9,
    )

    def model_function(_messages, info: AgentInfo):
        output_tool = info.output_tools[0]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tool.name,
                    {
                        "requirements": [],
                        "tasks": ["provide the POC plan by Friday"],
                        "decisions": [],
                        "risks": [],
                    },
                )
            ]
        )

    grounder = PydanticEvidenceGrounder(FunctionModel(model_function))
    original = extraction.model_copy(deep=True)
    grounded = grounder.ground(activity, extraction)
    assert extraction == original
    assert grounded.tasks[0].source_quote == "provide the POC plan by Friday"
    assert grounded.tasks[0].model_dump(exclude={"source_quote"}) == original.tasks[0].model_dump(
        exclude={"source_quote"}
    )


def test_evidence_grounder_rejects_non_verbatim_quote(session):
    activity = make_activity(session)
    extraction = StructuredActivityExtraction(
        summary="Follow up.",
        tasks=[TaskCandidate(title="Follow up", confidence=0.5)],
        overall_confidence=0.5,
    )

    def model_function(_messages, info: AgentInfo):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"requirements": [], "tasks": ["invented"], "decisions": [], "risks": []},
                )
            ]
        )

    with pytest.raises(EvidenceGroundingError, match="not present"):
        PydanticEvidenceGrounder(
            FunctionModel(model_function), validation_retries=0
        ).ground(activity, extraction)
    partial = PydanticEvidenceGrounder(
        FunctionModel(model_function), validation_retries=0, allow_partial=True
    ).ground(activity, extraction)
    assert partial.tasks[0].source_quote is None
    assert partial.review_required == extraction.review_required


def test_evidence_grounder_retries_semantic_validation(session):
    activity = make_activity(session)
    extraction = StructuredActivityExtraction(
        summary="Follow up.",
        tasks=[TaskCandidate(title="Follow up", confidence=0.5)],
        overall_confidence=0.5,
    )
    calls = 0

    def model_function(_messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        quote = "invented" if calls == 1 else "provide the POC plan by Friday"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"requirements": [], "tasks": [quote], "decisions": [], "risks": []},
                )
            ]
        )

    grounded = PydanticEvidenceGrounder(
        FunctionModel(model_function), validation_retries=1
    ).ground(activity, extraction)
    assert calls == 2
    assert grounded.tasks[0].source_quote == "provide the POC plan by Friday"
