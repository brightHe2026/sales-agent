from datetime import datetime, timezone

import pytest

from app.enums.memory import ActivityType, ExtractionStatus, ProjectStage, ProjectStatus, SourceType
from app.models import Activity, Customer, Project
from app.repositories.memory import ActivityRepository, ProjectMemoryRepository
from app.schemas.memory.activity import ActivityCreate, Participant
from app.services.activity_ingestion import (
    ActivityIngestionError,
    ActivityIngestionService,
    ActivityReferenceNotFoundError,
)


def payload(**overrides):
    values = {
        "activity_type": ActivityType.MEETING,
        "occurred_at": datetime.now(timezone.utc),
        "raw_content": "Customer needs a POC plan.",
        "source_type": SourceType.MANUAL,
    }
    values.update(overrides)
    return ActivityCreate(**values)


def test_ingests_unlinked_raw_activity(session):
    activity = ActivityIngestionService(ActivityRepository(session)).ingest(payload())
    persisted = session.get(Activity, activity.id)
    assert persisted is not None
    assert persisted.raw_content == "Customer needs a POC plan."
    assert persisted.extraction_status is ExtractionStatus.PENDING
    assert persisted.customer_id is None and persisted.project_id is None


def test_project_link_derives_its_customer_and_preserves_participants(session):
    customer = Customer(name="Acme")
    project = Project(customer=customer, name="Endpoint POC", stage=ProjectStage.DEMO_AND_POC, status=ProjectStatus.ACTIVE)
    session.add(project)
    session.commit()
    activity = ActivityIngestionService(ActivityRepository(session)).ingest(
        payload(project_id=project.id, participants=[Participant(name="Wang", role="Customer")])
    )
    assert activity.customer_id == customer.id
    assert activity.project_id == project.id
    assert activity.participants == [{"name": "Wang", "role": "Customer"}]


def test_rejects_mismatched_customer_and_project(session):
    project_customer = Customer(name="Acme")
    other_customer = Customer(name="Other")
    project = Project(customer=project_customer, name="POC", stage=ProjectStage.DEMO_AND_POC, status=ProjectStatus.ACTIVE)
    session.add_all([project, other_customer])
    session.commit()
    service = ActivityIngestionService(ActivityRepository(session))
    with pytest.raises(ActivityIngestionError, match="does not belong"):
        service.ingest(payload(project_id=project.id, customer_id=other_customer.id))
    assert session.query(Activity).count() == 0


def test_missing_project_returns_not_found_and_rolls_back(session):
    service = ActivityIngestionService(ActivityRepository(session))
    with pytest.raises(ActivityReferenceNotFoundError, match="Project not found"):
        service.ingest(payload(project_id="00000000-0000-0000-0000-000000000001"))
    assert session.query(Activity).count() == 0


def test_missing_customer_is_rejected(session):
    service = ActivityIngestionService(ActivityRepository(session))
    with pytest.raises(ActivityReferenceNotFoundError, match="Customer not found"):
        service.ingest(payload(customer_id="00000000-0000-0000-0000-000000000001"))
    assert session.query(Activity).count() == 0


def test_repository_create_rolls_back_commit_failure(session, monkeypatch):
    repository = ActivityRepository(session)
    activity = Activity(
        activity_type=ActivityType.MANUAL_NOTE,
        occurred_at=datetime.now(timezone.utc),
        raw_content="Must roll back",
        source_type=SourceType.MANUAL,
        extraction_status=ExtractionStatus.PENDING,
    )
    original_commit = session.commit

    def fail_commit():
        session.flush()
        raise RuntimeError("commit failed")

    monkeypatch.setattr(session, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="commit failed"):
        repository.create(activity)
    monkeypatch.setattr(session, "commit", original_commit)
    assert session.query(Activity).count() == 0


def test_project_memory_repository_loads_activity(session):
    customer = Customer(name="Acme")
    project = Project(customer=customer, name="POC", stage=ProjectStage.DEMO_AND_POC, status=ProjectStatus.ACTIVE)
    session.add(project)
    session.commit()
    ActivityIngestionService(ActivityRepository(session)).ingest(payload(project_id=project.id))
    memory = ProjectMemoryRepository(session).get(project.id)
    assert memory is not None
    assert [item.raw_content for item in memory.activities] == ["Customer needs a POC plan."]
    listed = ActivityRepository(session).list(project_id=project.id)
    assert [item.id for item in listed] == [memory.activities[0].id]
