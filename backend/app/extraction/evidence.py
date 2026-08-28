import json
from typing import Protocol

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.extraction.pydantic_ai import ActivityExtractor
from app.models import Activity
from app.schemas.memory.extraction import (
    FactEvidenceAssignments,
    StructuredActivityExtraction,
)


EVIDENCE_INSTRUCTIONS = """
For every supplied candidate fact, copy the shortest exact supporting substring from
raw_content. Preserve each list's order and length exactly. Never paraphrase, add, remove,
merge, split, reinterpret, or correct candidate facts. Return only substrings that occur
verbatim in raw_content.
""".strip()


class EvidenceGroundingError(ValueError):
    pass


class EvidenceGrounder(Protocol):
    def ground(
        self, activity: Activity, extraction: StructuredActivityExtraction
    ) -> StructuredActivityExtraction: ...


class PydanticEvidenceGrounder:
    version = "evidence-grounding-v1"

    def __init__(
        self,
        model: Model,
        *,
        retries: int = 2,
        validation_retries: int = 2,
        allow_partial: bool = False,
    ) -> None:
        self.agent = Agent(
            model,
            output_type=FactEvidenceAssignments,
            instructions=EVIDENCE_INSTRUCTIONS,
            retries=retries,
        )
        self.validation_retries = validation_retries
        self.allow_partial = allow_partial

    def ground(
        self, activity: Activity, extraction: StructuredActivityExtraction
    ) -> StructuredActivityExtraction:
        groups = ("requirements", "tasks", "decisions", "risks")
        candidates = {
            group: [
                item.model_dump(
                    mode="json", exclude={"source_quote", "confidence"}
                )
                for item in getattr(extraction, group)
            ]
            for group in groups
        }
        request = "Attach exact evidence to these immutable candidate facts:\n" + json.dumps(
            {"raw_content": activity.raw_content, "candidates": candidates},
            ensure_ascii=False,
        )
        last_error: EvidenceGroundingError | None = None
        for attempt in range(self.validation_retries + 1):
            prompt = request
            if last_error is not None:
                prompt += (
                    "\nThe previous response failed validation: "
                    f"{last_error}. Retry every list and copy only verbatim substrings."
                )
            result = self.agent.run_sync(prompt).output
            try:
                return self._apply_quotes(activity, extraction, result, groups)
            except EvidenceGroundingError as error:
                last_error = error
                if attempt == self.validation_retries:
                    if self.allow_partial:
                        return self._apply_partial_quotes(
                            activity, extraction, result, groups
                        )
                    raise
        raise AssertionError("unreachable")

    @staticmethod
    def _apply_quotes(activity, extraction, result, groups):
        grounded = extraction.model_copy(deep=True)
        for group in groups:
            facts = getattr(grounded, group)
            quotes = getattr(result, group)
            if len(quotes) != len(facts):
                raise EvidenceGroundingError(
                    f"Evidence count does not match {group} candidate count"
                )
            for fact, quote in zip(facts, quotes, strict=True):
                if quote not in activity.raw_content:
                    raise EvidenceGroundingError(
                        f"Evidence quote for {group} {fact.title!r} is not present in raw content"
                    )
                fact.source_quote = quote
        return grounded

    @staticmethod
    def _apply_partial_quotes(activity, extraction, result, groups):
        grounded = extraction.model_copy(deep=True)
        for group in groups:
            facts = getattr(grounded, group)
            quotes = getattr(result, group)
            for fact, quote in zip(facts, quotes):
                fact.source_quote = quote if quote in activity.raw_content else None
        return grounded


class GroundedActivityExtractor:
    def __init__(self, extractor: ActivityExtractor, grounder: EvidenceGrounder) -> None:
        self.extractor = extractor
        self.grounder = grounder
        self.prompt_version = (
            f"{getattr(extractor, 'prompt_version', 'unknown')}+"
            f"{getattr(grounder, 'version', 'unknown')}"
        )

    def extract(self, activity: Activity) -> StructuredActivityExtraction:
        extraction = self.extractor.extract(activity)
        return self.grounder.ground(activity, extraction)
