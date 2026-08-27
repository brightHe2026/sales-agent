import uuid

from pydantic import BaseModel

from app.enums.memory import ExtractionStatus
from app.schemas.memory.extraction import StructuredActivityExtraction
from app.schemas.memory.update import MemoryUpdateResult


class ActivityProcessingResult(BaseModel):
    activity_id: uuid.UUID
    extraction_status: ExtractionStatus
    extraction: StructuredActivityExtraction
    memory_update: MemoryUpdateResult | None = None
