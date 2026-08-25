import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_session
from app.repositories.memory import ActivityRepository
from app.schemas.memory.activity import ActivityCreate, ActivityRead
from app.services.activity_ingestion import (
    ActivityIngestionError,
    ActivityIngestionService,
    ActivityReferenceNotFoundError,
)


router = APIRouter(prefix="/activities", tags=["activities"])
DatabaseSession = Annotated[Session, Depends(get_session)]


@router.post("", response_model=ActivityRead, status_code=status.HTTP_201_CREATED)
def ingest_activity(payload: ActivityCreate, session: DatabaseSession) -> ActivityRead:
    service = ActivityIngestionService(ActivityRepository(session))
    try:
        activity = service.ingest(payload)
    except ActivityReferenceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ActivityIngestionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return ActivityRead.model_validate(activity)


@router.get("/{activity_id}", response_model=ActivityRead)
def get_activity(activity_id: uuid.UUID, session: DatabaseSession) -> ActivityRead:
    repository = ActivityRepository(session)
    activity = repository.get(activity_id)
    if activity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Activity not found")
    return ActivityRead.model_validate(activity)
