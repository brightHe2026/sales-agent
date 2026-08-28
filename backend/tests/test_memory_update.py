import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.dialects import postgresql

from app.enums.memory import (
    ActivityType,
    ExtractionStatus,
    OwnerType,
    ProjectStage,
    ProjectStatus,
    RiskSeverity,
    SourceType,
)
from app.models import Activity, Customer, Decision, Project, Requirement, Risk, Task
from app.repositories import ActivityRepository, MemoryUpdateRepository
from app.schemas.memory.extraction import (
    DecisionCandidate,
    RequirementCandidate,
    RiskCandidate,
    StructuredActivityExtraction,
    TaskCandidate,
)
from app.services.memory_update import (
    CandidateValidationError,
    MemoryUpdateService,
    candidate_fingerprint,
)


def setup_activity(session, *, linked=True, status=ExtractionStatus.PROCESSED):
    project = None
    if linked:
        customer = Customer(name="Acme")
        project = Project(
            customer=customer,
            name="Endpoint POC",
            stage=ProjectStage.DEMO_AND_POC,
            status=ProjectStatus.ACTIVE,
        )
        session.add(project)
        session.commit()
    activity = Activity(
        project_id=project.id if project else None,
        customer_id=project.customer_id if project else None,
        activity_type=ActivityType.MEETING,
        occurred_at=datetime.now(timezone.utc),
        raw_content="Original evidence",
        source_type=SourceType.MANUAL,
        extraction_status=status,
        extraction_version="task003-v1",
    )
    return ActivityRepository(session).create(activity), project


def extraction(*, review_required=False):
    return StructuredActivityExtraction(
        summary="Structured facts",
        requirements=[RequirementCandidate(title="Support SSO", description="SAML SSO is required", source_quote="Original evidence", confidence=0.9)],
        tasks=[TaskCandidate(title="Send POC plan", owner_type=OwnerType.SELF, source_quote="Original evidence", confidence=0.8)],
        decisions=[DecisionCandidate(title="Run a POC", description="Customer agreed to run a POC", source_quote="Original evidence", confidence=0.85)],
        risks=[RiskCandidate(title="Environment unavailable", description="Test environment is not ready", source_quote="Original evidence", severity=RiskSeverity.HIGH, confidence=0.7)],
        overall_confidence=0.8,
        review_required=review_required,
    )


def service(session):
    return MemoryUpdateService(MemoryUpdateRepository(session))


def test_applies_all_candidate_types_atomically_and_preserves_sources(session):
    activity, project = setup_activity(session)
    original_stage = project.stage
    result = service(session).apply(activity.id, extraction())
    assert result.requirements.created == 1
    assert result.tasks.created == 1
    assert result.decisions.created == 1
    assert result.risks.created == 1
    for model in (Requirement, Task, Decision, Risk):
        fact = session.query(model).one()
        assert fact.project_id == project.id
        assert fact.source_activity_id == activity.id
        assert len(fact.source_fingerprint) == 64
        assert fact.source_quote == "Original evidence"
    session.refresh(activity)
    session.refresh(project)
    assert activity.raw_content == "Original evidence"
    assert project.stage is original_stage


def test_reapplying_same_extraction_is_idempotent(session):
    activity, _ = setup_activity(session)
    updater = service(session)
    updater.apply(activity.id, extraction())
    second = updater.apply(activity.id, extraction())
    assert second.requirements.skipped == 1
    assert second.tasks.skipped == 1
    assert second.decisions.skipped == 1
    assert second.risks.skipped == 1
    assert session.query(Requirement).count() == 1
    assert session.query(Task).count() == 1
    assert session.query(Decision).count() == 1
    assert session.query(Risk).count() == 1


def test_duplicate_candidates_in_one_extraction_are_skipped(session):
    activity, _ = setup_activity(session)
    duplicate = extraction()
    duplicate.requirements.append(duplicate.requirements[0].model_copy(deep=True))
    result = service(session).apply(activity.id, duplicate)
    assert result.requirements.created == 1
    assert result.requirements.skipped == 1
    assert session.query(Requirement).count() == 1


def test_activity_lock_uses_postgresql_for_update(session, monkeypatch):
    captured = {}

    def capture(statement):
        captured["statement"] = statement
        return None

    monkeypatch.setattr(session, "scalar", capture)
    MemoryUpdateRepository(session).lock_activity(uuid.uuid4())
    sql = str(captured["statement"].compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in sql


@pytest.mark.parametrize(
    ("linked", "status", "review_required", "message"),
    [
        (False, ExtractionStatus.PROCESSED, False, "linked to a project"),
        (True, ExtractionStatus.PENDING, False, "must be processed"),
        (True, ExtractionStatus.PROCESSED, True, "human review"),
    ],
)
def test_invalid_update_gates_write_nothing(session, linked, status, review_required, message):
    activity, _ = setup_activity(session, linked=linked, status=status)
    with pytest.raises(CandidateValidationError, match=message):
        service(session).apply(activity.id, extraction(review_required=review_required))
    assert session.query(Requirement).count() == 0
    assert session.query(Task).count() == 0
    assert session.query(Decision).count() == 0
    assert session.query(Risk).count() == 0


def test_incomplete_required_description_gates_entire_batch(session):
    activity, _ = setup_activity(session)
    invalid = extraction()
    invalid.requirements[0].description = "   "
    with pytest.raises(CandidateValidationError, match="descriptions are required"):
        service(session).apply(activity.id, invalid)
    assert session.query(Task).count() == 0


def test_unknown_owner_with_name_is_rejected(session):
    activity, _ = setup_activity(session)
    invalid = extraction()
    invalid.tasks[0].owner_type = OwnerType.UNKNOWN
    invalid.tasks[0].owner_name = "Guessed Person"
    with pytest.raises(CandidateValidationError, match="Unknown task owner"):
        service(session).apply(activity.id, invalid)
    assert session.query(Task).count() == 0


@pytest.mark.parametrize("source_quote", [None, "Invented evidence"])
def test_missing_or_ungrounded_source_quote_gates_entire_batch(session, source_quote):
    activity, _ = setup_activity(session)
    invalid = extraction()
    invalid.tasks[0].source_quote = source_quote
    with pytest.raises(CandidateValidationError, match="source quote"):
        service(session).apply(activity.id, invalid)
    assert session.query(Requirement).count() == 0
    assert session.query(Task).count() == 0


def test_source_quote_does_not_change_semantic_fingerprint():
    first = TaskCandidate(title="Send POC plan", source_quote="First quote", confidence=0.8)
    second = TaskCandidate(title="Send POC plan", source_quote="Second quote", confidence=0.8)
    assert candidate_fingerprint("task", first) == candidate_fingerprint("task", second)


def test_repository_rolls_back_failed_atomic_write(session, monkeypatch):
    activity, _ = setup_activity(session)
    updater = service(session)
    original_commit = session.commit

    def fail_commit():
        session.flush()
        raise RuntimeError("commit failed")

    monkeypatch.setattr(session, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="commit failed"):
        updater.apply(activity.id, extraction())
    monkeypatch.setattr(session, "commit", original_commit)
    assert session.query(Requirement).count() == 0
    assert session.query(Task).count() == 0
    assert session.query(Decision).count() == 0
    assert session.query(Risk).count() == 0
