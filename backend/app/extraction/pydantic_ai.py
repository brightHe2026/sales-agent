import json
from typing import Protocol

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.models import Activity
from app.schemas.memory.extraction import StructuredActivityExtraction


EXTRACTION_INSTRUCTIONS = """
Extract only facts explicitly supported by the activity. Never invent owners,
dates, customer names, project names, decisions, requirements, or risks.
Use UNKNOWN or null when evidence is insufficient. Candidate facts are proposals
for later validation; they must not imply that anything was written to a database.
Set review_required when the source is ambiguous, contradictory, or confidence is low.
""".strip()


class ActivityExtractor(Protocol):
    def extract(self, activity: Activity) -> StructuredActivityExtraction: ...


class PydanticActivityExtractor:
    def __init__(self, model: Model, *, retries: int = 2) -> None:
        self.agent = Agent(
            model,
            output_type=StructuredActivityExtraction,
            instructions=EXTRACTION_INSTRUCTIONS,
            retries=retries,
        )

    def extract(self, activity: Activity) -> StructuredActivityExtraction:
        source = {
            "activity_type": activity.activity_type.value,
            "occurred_at": activity.occurred_at.isoformat(),
            "raw_content": activity.raw_content,
            "participants": activity.participants,
            "customer_id": str(activity.customer_id) if activity.customer_id else None,
            "project_id": str(activity.project_id) if activity.project_id else None,
        }
        result = self.agent.run_sync(
            "Extract structured candidate facts from this activity:\n"
            + json.dumps(source, ensure_ascii=False)
        )
        return result.output
