from datetime import datetime, timezone

import pytest

from app.enums.memory import (
    ActivityType,
    ExtractionStatus,
    OwnerType,
    ProjectStage,
    ProjectStatus,
    SourceType,
)
from app.models import Activity, Customer, Project, Requirement, Task
from app.repositories import ActivityRepository, MemoryUpdateRepository
from app.schemas.memory.activity import ActivityCreate
from app.schemas.memory.extraction import (
    RequirementCandidate,
    StructuredActivityExtraction,
    TaskCandidate,
)
from app.services.activity_extraction import ActivityExtractionError, ActivityExtractionService
from app.services.activity_ingestion import ActivityIngestionService
from app.services.activity_processing import ActivityProcessingPipeline
from app.services.memory_update import CandidateValidationError, MemoryUpdateService


class StubExtractor:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def extract(self, activity):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def setup_project(session):
    customer = Customer(name="Acme")
    project = Project(
        customer=customer,
        name="Endpoint POC",
        stage=ProjectStage.DEMO_AND_POC,
        status=ProjectStatus.ACTIVE,
    )
    session.add(project)
    session.commit()
    return customer, project


def payload(project):
    return ActivityCreate(
        customer_id=project.customer_id,
        project_id=project.id,
        activity_type=ActivityType.MEETING,
        occurred_at=datetime.now(timezone.utc),
        raw_content="Customer requires SSO. I will send a POC plan.",
        source_type=SourceType.MANUAL,
    )


def extraction(*, review_required=False, requirement_description="SAML SSO is required"):
    return StructuredActivityExtraction(
        summary="Customer requires SSO and expects a POC plan.",
        requirements=[
            RequirementCandidate(
                title="Support SSO",
                description=requirement_description,
                confidence=0.9,
            )
        ],
        tasks=[
            TaskCandidate(
                title="Send POC plan",
                owner_type=OwnerType.SELF,
                confidence=0.85,
            )
        ],
        overall_confidence=0.88,
        review_required=review_required,
    )


def pipeline(session, extractor):
    activity_repository = ActivityRepository(session)
    return ActivityProcessingPipeline(
        ActivityIngestionService(activity_repository),
        ActivityExtractionService(
            activity_repository,
            extractor,
            extraction_version="task005-v1",
        ),
        MemoryUpdateService(MemoryUpdateRepository(session)),
    )


def test_ingest_and_process_completes_full_memory_pipeline(session):
    _, project = setup_project(session)
    extractor = StubExtractor(result=extraction())
    result = pipeline(session, extractor).ingest_and_process(payload(project))
    activity = session.get(Activity, result.activity_id)
    requirement = session.query(Requirement).one()
    task = session.query(Task).one()
    assert extractor.calls == 1
    assert result.extraction_status is ExtractionStatus.PROCESSED
    assert result.memory_update.requirements.created == 1
    assert result.memory_update.tasks.created == 1
    assert activity.raw_content == "Customer requires SSO. I will send a POC plan."
    assert activity.summary == result.extraction.summary
    assert requirement.source_activity_id == activity.id
    assert task.source_activity_id == activity.id


def test_review_required_stops_before_memory_update(session):
    _, project = setup_project(session)
    result = pipeline(session, StubExtractor(result=extraction(review_required=True))).ingest_and_process(
        payload(project)
    )
    activity = session.get(Activity, result.activity_id)
    assert result.extraction_status is ExtractionStatus.REVIEW_REQUIRED
    assert result.memory_update is None
    assert activity.extraction_status is ExtractionStatus.REVIEW_REQUIRED
    assert session.query(Requirement).count() == 0
    assert session.query(Task).count() == 0


def test_extraction_failure_leaves_durable_failed_activity(session):
    _, project = setup_project(session)
    with pytest.raises(ActivityExtractionError):
        pipeline(session, StubExtractor(error=TimeoutError("model timeout"))).ingest_and_process(
            payload(project)
        )
    activity = session.query(Activity).one()
    assert activity.extraction_status is ExtractionStatus.FAILED
    assert activity.raw_content == "Customer requires SSO. I will send a POC plan."
    assert session.query(Requirement).count() == 0


def test_validation_failure_keeps_processed_activity_for_retry(session):
    _, project = setup_project(session)
    invalid = extraction(requirement_description="   ")
    with pytest.raises(CandidateValidationError):
        pipeline(session, StubExtractor(result=invalid)).ingest_and_process(payload(project))
    activity = session.query(Activity).one()
    assert activity.extraction_status is ExtractionStatus.PROCESSED
    assert activity.summary == invalid.summary
    assert activity.raw_content == "Customer requires SSO. I will send a POC plan."
    assert session.query(Requirement).count() == 0
    assert session.query(Task).count() == 0


def test_reprocessing_existing_activity_is_idempotent(session):
    _, project = setup_project(session)
    extractor = StubExtractor(result=extraction())
    processor = pipeline(session, extractor)
    first = processor.ingest_and_process(payload(project))
    second = processor.process_existing(first.activity_id)
    assert second.memory_update.requirements.skipped == 1
    assert second.memory_update.tasks.skipped == 1
    assert session.query(Requirement).count() == 1
    assert session.query(Task).count() == 1
