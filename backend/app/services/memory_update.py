import hashlib
import json
import uuid

from pydantic import BaseModel

from app.enums.memory import ExtractionStatus, RequirementStatus, RiskStatus, TaskStatus
from app.models import Decision, Requirement, Risk, Task
from app.repositories.memory_update import MemoryUpdateRepository
from app.schemas.memory.extraction import StructuredActivityExtraction
from app.schemas.memory.update import FactUpdateCount, MemoryUpdateResult


class CandidateValidationError(ValueError):
    pass


def candidate_fingerprint(kind: str, candidate: BaseModel) -> str:
    semantic_data = candidate.model_dump(mode="json", exclude={"confidence"})
    canonical = json.dumps(semantic_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{kind}:{canonical}".encode()).hexdigest()


class MemoryUpdateService:
    def __init__(
        self,
        update_repository: MemoryUpdateRepository,
    ) -> None:
        self.update_repository = update_repository

    def apply(
        self,
        activity_id: uuid.UUID,
        extraction: StructuredActivityExtraction,
    ) -> MemoryUpdateResult:
        activity = self.update_repository.lock_activity(activity_id)
        try:
            if activity is None:
                raise CandidateValidationError("Activity not found")
            if activity.project_id is None:
                raise CandidateValidationError("Activity must be linked to a project")
            if activity.extraction_status is not ExtractionStatus.PROCESSED:
                raise CandidateValidationError("Activity extraction must be processed")
            if extraction.review_required:
                raise CandidateValidationError("Extraction requires human review")
            self._validate_required_descriptions(extraction)
            self._validate_task_owners(extraction)
        except CandidateValidationError:
            self.update_repository.rollback()
            raise

        groups = {
            "requirements": self._requirements(activity.id, activity.project_id, extraction),
            "tasks": self._tasks(activity.id, activity.project_id, extraction),
            "decisions": self._decisions(activity.id, activity.project_id, extraction),
            "risks": self._risks(activity.id, activity.project_id, extraction),
        }
        models = {
            "requirements": Requirement,
            "tasks": Task,
            "decisions": Decision,
            "risks": Risk,
        }
        counts: dict[str, FactUpdateCount] = {}
        new_facts = []
        for name, facts in groups.items():
            unique_facts = {
                fact.source_fingerprint: fact
                for fact in facts
                if fact.source_fingerprint is not None
            }
            fingerprints = set(unique_facts)
            existing = self.update_repository.existing_fingerprints(
                models[name], activity.id, fingerprints
            )
            pending = [
                fact for fingerprint, fact in unique_facts.items() if fingerprint not in existing
            ]
            new_facts.extend(pending)
            counts[name] = FactUpdateCount(created=len(pending), skipped=len(facts) - len(pending))

        self.update_repository.create_all(new_facts)
        return MemoryUpdateResult(**counts)

    @staticmethod
    def _validate_required_descriptions(extraction: StructuredActivityExtraction) -> None:
        candidates = [*extraction.requirements, *extraction.decisions, *extraction.risks]
        if any(not candidate.description or not candidate.description.strip() for candidate in candidates):
            raise CandidateValidationError("Requirement, decision, and risk descriptions are required")

    @staticmethod
    def _validate_task_owners(extraction: StructuredActivityExtraction) -> None:
        if any(task.owner_type.value == "UNKNOWN" and task.owner_name for task in extraction.tasks):
            raise CandidateValidationError("Unknown task owner cannot have an owner name")

    @staticmethod
    def _requirements(activity_id, project_id, extraction):
        return [
            Requirement(
                project_id=project_id,
                source_activity_id=activity_id,
                source_fingerprint=candidate_fingerprint("requirement", item),
                title=item.title,
                description=item.description,
                requirement_type=item.requirement_type,
                priority=item.priority,
                status=RequirementStatus.OPEN,
                confidence=item.confidence,
            )
            for item in extraction.requirements
        ]

    @staticmethod
    def _tasks(activity_id, project_id, extraction):
        return [
            Task(
                project_id=project_id,
                source_activity_id=activity_id,
                source_fingerprint=candidate_fingerprint("task", item),
                title=item.title,
                description=item.description,
                owner_type=item.owner_type,
                owner_name=item.owner_name,
                due_at=item.due_at,
                priority=item.priority,
                status=TaskStatus.OPEN,
                confidence=item.confidence,
            )
            for item in extraction.tasks
        ]

    @staticmethod
    def _decisions(activity_id, project_id, extraction):
        return [
            Decision(
                project_id=project_id,
                source_activity_id=activity_id,
                source_fingerprint=candidate_fingerprint("decision", item),
                title=item.title,
                description=item.description,
                decision_maker=item.decision_maker,
                decided_at=item.decided_at,
                confidence=item.confidence,
            )
            for item in extraction.decisions
        ]

    @staticmethod
    def _risks(activity_id, project_id, extraction):
        return [
            Risk(
                project_id=project_id,
                source_activity_id=activity_id,
                source_fingerprint=candidate_fingerprint("risk", item),
                title=item.title,
                description=item.description,
                severity=item.severity,
                status=RiskStatus.OPEN,
                mitigation=item.mitigation,
                confidence=item.confidence,
            )
            for item in extraction.risks
        ]
