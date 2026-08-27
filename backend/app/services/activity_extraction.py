import uuid

from app.enums.memory import ExtractionStatus
from app.extraction import ActivityExtractor
from app.repositories.memory import ActivityRepository
from app.schemas.memory.extraction import StructuredActivityExtraction


class ActivityExtractionError(RuntimeError):
    pass


class ActivityForExtractionNotFoundError(ActivityExtractionError):
    pass


class ActivityExtractionService:
    def __init__(
        self,
        repository: ActivityRepository,
        extractor: ActivityExtractor,
        *,
        extraction_version: str,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.extraction_version = extraction_version

    def extract(self, activity_id: uuid.UUID) -> StructuredActivityExtraction:
        activity = self.repository.get(activity_id)
        if activity is None:
            raise ActivityForExtractionNotFoundError("Activity not found")

        try:
            result = self.extractor.extract(activity)
        except Exception as exc:
            activity.extraction_status = ExtractionStatus.FAILED
            activity.extraction_version = self.extraction_version
            activity.extraction_confidence = None
            self.repository.save(activity)
            raise ActivityExtractionError("Activity extraction failed") from exc

        activity.summary = result.summary
        activity.extraction_status = (
            ExtractionStatus.REVIEW_REQUIRED
            if result.review_required
            else ExtractionStatus.PROCESSED
        )
        activity.extraction_version = self.extraction_version
        activity.extraction_confidence = result.overall_confidence
        self.repository.save(activity)
        return result
