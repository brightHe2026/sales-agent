import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Activity, Customer, Project


class ActivityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, activity_id: uuid.UUID | str) -> Activity | None:
        try:
            parsed_id = uuid.UUID(str(activity_id))
        except ValueError:
            return None
        return self.session.get(Activity, parsed_id)

    def get_customer(self, customer_id: uuid.UUID) -> Customer | None:
        return self.session.get(Customer, customer_id)

    def get_project(self, project_id: uuid.UUID) -> Project | None:
        return self.session.get(Project, project_id)

    def list(
        self,
        *,
        project_id: uuid.UUID | None = None,
        customer_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> list[Activity]:
        statement = select(Activity).order_by(Activity.occurred_at.desc()).limit(limit)
        if project_id is not None:
            statement = statement.where(Activity.project_id == project_id)
        if customer_id is not None:
            statement = statement.where(Activity.customer_id == customer_id)
        return list(self.session.scalars(statement))

    def create(self, activity: Activity) -> Activity:
        try:
            self.session.add(activity)
            self.session.commit()
            return activity
        except Exception:
            self.session.rollback()
            raise


class ProjectMemoryRepository:
    """Read a project and its current structured memory in one repository call."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, project_id: uuid.UUID) -> Project | None:
        statement = (
            select(Project)
            .where(Project.id == project_id)
            .options(
                selectinload(Project.activities),
                selectinload(Project.requirements),
                selectinload(Project.tasks),
                selectinload(Project.decisions),
                selectinload(Project.risks),
            )
        )
        return self.session.scalar(statement)
