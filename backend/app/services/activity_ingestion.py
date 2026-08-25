from app.enums.memory import ExtractionStatus
from app.models import Activity
from app.repositories.memory import ActivityRepository
from app.schemas.memory.activity import ActivityCreate


class ActivityIngestionError(ValueError):
    pass


class ActivityReferenceNotFoundError(ActivityIngestionError):
    pass


class ActivityIngestionService:
    def __init__(self, repository: ActivityRepository) -> None:
        self.repository = repository

    def ingest(self, payload: ActivityCreate) -> Activity:
        customer = self.repository.get_customer(payload.customer_id) if payload.customer_id else None
        if payload.customer_id and customer is None:
            raise ActivityReferenceNotFoundError("Customer not found")

        project = self.repository.get_project(payload.project_id) if payload.project_id else None
        if payload.project_id and project is None:
            raise ActivityReferenceNotFoundError("Project not found")
        if project and customer and project.customer_id != customer.id:
            raise ActivityIngestionError("Project does not belong to the supplied customer")

        customer_id = customer.id if customer else (project.customer_id if project else None)
        activity = Activity(
            customer_id=customer_id,
            project_id=project.id if project else None,
            activity_type=payload.activity_type,
            occurred_at=payload.occurred_at,
            raw_content=payload.raw_content,
            summary=payload.summary,
            source_type=payload.source_type,
            source_ref=payload.source_ref,
            participants=[participant.model_dump() for participant in payload.participants]
            if payload.participants is not None
            else None,
            extraction_status=ExtractionStatus.PENDING,
        )
        return self.repository.create(activity)
