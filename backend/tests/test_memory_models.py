from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.enums.memory import ActivityType, ExtractionStatus, OwnerType, ProjectStage, ProjectStatus, RequirementStatus, SourceType, TaskStatus
from app.models import Activity, Customer, Project, Requirement, Task


def make_project(session):
    customer = Customer(name="Acme")
    project = Project(customer=customer, name="POC", stage=ProjectStage.DEMO_AND_POC, status=ProjectStatus.ACTIVE)
    session.add(project)
    session.commit()
    return customer, project


def test_activity_can_store_raw_evidence_without_customer_or_project(session):
    activity = Activity(activity_type=ActivityType.MANUAL_NOTE, occurred_at=datetime.now(timezone.utc), raw_content="Original evidence", source_type=SourceType.MANUAL, extraction_status=ExtractionStatus.PENDING)
    session.add(activity)
    session.commit()
    assert activity.raw_content == "Original evidence"
    assert activity.customer_id is None and activity.project_id is None


def test_activity_raw_content_is_required(session):
    session.add(Activity(activity_type=ActivityType.MEETING, occurred_at=datetime.now(timezone.utc), raw_content=None, source_type=SourceType.MANUAL, extraction_status=ExtractionStatus.PENDING))  # type: ignore[arg-type]
    with pytest.raises(IntegrityError):
        session.commit()


def test_derived_facts_trace_to_source_activity(session):
    _, project = make_project(session)
    activity = Activity(project=project, activity_type=ActivityType.MEETING, occurred_at=datetime.now(timezone.utc), raw_content="Need SSO", source_type=SourceType.MANUAL, extraction_status=ExtractionStatus.PROCESSED)
    requirement = Requirement(project=project, source_activity_id=activity.id, title="SSO", description="Support SAML", status=RequirementStatus.CONFIRMED, confidence=1.0)
    session.add_all([activity, requirement])
    session.flush()
    requirement.source_activity_id = activity.id
    session.commit()
    assert requirement.source_activity_id == activity.id


def test_deleting_project_preserves_activity_and_raw_content(session):
    _, project = make_project(session)
    activity = Activity(project=project, activity_type=ActivityType.MEETING, occurred_at=datetime.now(timezone.utc), raw_content="Historical fact", source_type=SourceType.MANUAL, extraction_status=ExtractionStatus.PENDING)
    session.add(activity)
    session.commit()
    activity_id = activity.id
    session.delete(project)
    session.commit()
    session.expire_all()
    preserved = session.get(Activity, activity_id)
    assert preserved is not None
    assert preserved.project_id is None
    assert preserved.raw_content == "Historical fact"


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_confidence_boundaries_are_accepted(session, value):
    _, project = make_project(session)
    session.add(Task(project=project, title="Follow up", owner_type=OwnerType.UNKNOWN, status=TaskStatus.OPEN, confidence=value))
    session.commit()


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_confidence_outside_range_is_rejected(session, value):
    _, project = make_project(session)
    session.add(Task(project=project, title="Follow up", owner_type=OwnerType.UNKNOWN, status=TaskStatus.OPEN, confidence=value))
    with pytest.raises(IntegrityError):
        session.commit()
