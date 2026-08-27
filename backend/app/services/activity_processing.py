import uuid

from app.enums.memory import ExtractionStatus
from app.schemas.memory.activity import ActivityCreate
from app.schemas.memory.processing import ActivityProcessingResult
from app.services.activity_extraction import ActivityExtractionService
from app.services.activity_ingestion import ActivityIngestionService
from app.services.memory_update import MemoryUpdateService


class ActivityProcessingPipeline:
    """Coordinate durable POC-01 processing stages without hiding their failures."""

    def __init__(
        self,
        ingestion_service: ActivityIngestionService,
        extraction_service: ActivityExtractionService,
        memory_update_service: MemoryUpdateService,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.extraction_service = extraction_service
        self.memory_update_service = memory_update_service

    def ingest_and_process(self, payload: ActivityCreate) -> ActivityProcessingResult:
        activity = self.ingestion_service.ingest(payload)
        return self.process_existing(activity.id)

    def process_existing(self, activity_id: uuid.UUID) -> ActivityProcessingResult:
        extraction = self.extraction_service.extract(activity_id)
        if extraction.review_required:
            return ActivityProcessingResult(
                activity_id=activity_id,
                extraction_status=ExtractionStatus.REVIEW_REQUIRED,
                extraction=extraction,
            )

        memory_update = self.memory_update_service.apply(activity_id, extraction)
        return ActivityProcessingResult(
            activity_id=activity_id,
            extraction_status=ExtractionStatus.PROCESSED,
            extraction=extraction,
            memory_update=memory_update,
        )
