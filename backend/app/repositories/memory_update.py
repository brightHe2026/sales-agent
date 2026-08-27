import uuid
from collections.abc import Sequence
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Activity, Decision, Requirement, Risk, Task


Fact = Requirement | Task | Decision | Risk
FactType = TypeVar("FactType", Requirement, Task, Decision, Risk)


class MemoryUpdateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def lock_activity(self, activity_id: uuid.UUID) -> Activity | None:
        statement = (
            select(Activity)
            .where(Activity.id == activity_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return self.session.scalar(statement)

    def existing_fingerprints(
        self,
        model: type[FactType],
        source_activity_id: uuid.UUID,
        fingerprints: set[str],
    ) -> set[str]:
        if not fingerprints:
            return set()
        statement = select(model.source_fingerprint).where(
            model.source_activity_id == source_activity_id,
            model.source_fingerprint.in_(fingerprints),
        )
        return {value for value in self.session.scalars(statement) if value is not None}

    def create_all(self, facts: Sequence[Fact]) -> None:
        try:
            self.session.add_all(facts)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    def rollback(self) -> None:
        self.session.rollback()
