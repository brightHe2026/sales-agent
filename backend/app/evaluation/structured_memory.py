from collections import Counter
from pathlib import Path

from app.enums.memory import ExtractionStatus
from app.extraction import ActivityExtractor
from app.models import Activity
from app.schemas.evaluation import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationReport,
    EvaluationThresholds,
)


def load_dataset(path: str | Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


class StructuredMemoryEvaluator:
    def __init__(
        self,
        extractor: ActivityExtractor,
        thresholds: EvaluationThresholds | None = None,
    ) -> None:
        self.extractor = extractor
        self.thresholds = thresholds or EvaluationThresholds()

    def evaluate(self, dataset: EvaluationDataset) -> EvaluationReport:
        cases = [self._evaluate_case(case) for case in dataset.cases]
        expected_total = sum(case.expected_facts for case in cases)
        actual_total = sum(case.actual_facts for case in cases)
        matched_total = sum(case.matched_facts for case in cases)
        owner_total = sum(case.owner_checks for case in cases)
        owner_matches = sum(case.owner_matches for case in cases)
        precision = matched_total / actual_total if actual_total else float(expected_total == 0)
        recall = matched_total / expected_total if expected_total else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        owner_accuracy = owner_matches / owner_total if owner_total else 1.0
        review_accuracy = sum(case.review_required_match for case in cases) / len(cases)
        hallucinations = sum(case.hallucinated_facts for case in cases)
        passed = (
            precision >= self.thresholds.min_precision
            and recall >= self.thresholds.min_recall
            and owner_accuracy >= self.thresholds.min_owner_accuracy
            and review_accuracy >= self.thresholds.min_review_accuracy
            and hallucinations <= self.thresholds.max_hallucinated_facts
        )
        return EvaluationReport(
            dataset_name=dataset.name,
            case_count=len(cases),
            precision=precision,
            recall=recall,
            f1=f1,
            owner_accuracy=owner_accuracy,
            review_required_accuracy=review_accuracy,
            hallucinated_facts=hallucinations,
            passed=passed,
            cases=cases,
        )

    def _evaluate_case(self, case: EvaluationCase) -> EvaluationCaseResult:
        activity = Activity(
            activity_type=case.activity_type,
            occurred_at=case.occurred_at,
            raw_content=case.raw_content,
            source_type=case.source_type,
            participants=[participant.model_dump() for participant in case.participants]
            if case.participants is not None
            else None,
            extraction_status=ExtractionStatus.PENDING,
        )
        actual = self.extractor.extract(activity)
        expected_by_kind = self._fact_titles(case.expected)
        actual_by_kind = self._fact_titles(actual)
        matched = 0
        expected_count = 0
        actual_count = 0
        missing: list[str] = []
        hallucinated: list[str] = []
        task_title_pairs: list[tuple[str, str]] = []
        for kind in expected_by_kind:
            expected = expected_by_kind[kind]
            observed = actual_by_kind[kind]
            pairs, missing_facts, hallucinated_facts = self._match_facts(
                kind,
                expected,
                observed,
                case.fact_aliases,
            )
            matched += len(pairs)
            expected_count += expected.total()
            actual_count += observed.total()
            missing.extend(self._labels(kind, missing_facts))
            hallucinated.extend(self._labels(kind, hallucinated_facts))
            if kind == "task":
                task_title_pairs = pairs

        expected_tasks = {normalize(item.title): item for item in case.expected.tasks}
        actual_tasks = {normalize(item.title): item for item in actual.tasks}
        owner_matches = sum(
            expected_tasks[expected_title].owner_type == actual_tasks[actual_title].owner_type
            and normalize(expected_tasks[expected_title].owner_name or "")
            == normalize(actual_tasks[actual_title].owner_name or "")
            for expected_title, actual_title in task_title_pairs
        )
        return EvaluationCaseResult(
            case_id=case.id,
            expected_facts=expected_count,
            actual_facts=actual_count,
            matched_facts=matched,
            hallucinated_facts=len(hallucinated),
            owner_checks=len(expected_tasks),
            owner_matches=owner_matches,
            review_required_match=actual.review_required == case.expected.review_required,
            missing_facts=missing,
            hallucinated_fact_labels=hallucinated,
        )

    @staticmethod
    def _fact_titles(extraction):
        project_signal = extraction.project_signal
        return {
            "project": Counter(
                [normalize(project_signal.project_name)]
                if project_signal and project_signal.project_name
                else []
            ),
            "customer": Counter(
                [normalize(project_signal.customer_name)]
                if project_signal and project_signal.customer_name
                else []
            ),
            "requirement": Counter(normalize(item.title) for item in extraction.requirements),
            "task": Counter(normalize(item.title) for item in extraction.tasks),
            "decision": Counter(normalize(item.title) for item in extraction.decisions),
            "risk": Counter(normalize(item.title) for item in extraction.risks),
        }

    @staticmethod
    def _labels(kind: str, facts: Counter[str]) -> list[str]:
        return [
            f"{kind}:{title}"
            for title in sorted(facts)
            for _ in range(facts[title])
        ]

    def _match_facts(
        self,
        kind: str,
        expected: Counter[str],
        actual: Counter[str],
        aliases: dict[str, list[str]],
    ) -> tuple[list[tuple[str, str]], Counter[str], Counter[str]]:
        expected_items = list(expected.elements())
        actual_items = list(actual.elements())
        pairs: list[tuple[str, str]] = []
        matched_expected: set[int] = set()
        matched_actual: set[int] = set()
        for expected_index, expected_title in enumerate(expected_items):
            accepted = {expected_title}
            accepted.update(
                normalize(alias)
                for alias in aliases.get(f"{kind}:{expected_title}", [])
            )
            for actual_index, actual_title in enumerate(actual_items):
                if actual_index not in matched_actual and actual_title in accepted:
                    matched_expected.add(expected_index)
                    matched_actual.add(actual_index)
                    pairs.append((expected_title, actual_title))
                    break
        missing = Counter(
            title for index, title in enumerate(expected_items) if index not in matched_expected
        )
        hallucinated = Counter(
            title for index, title in enumerate(actual_items) if index not in matched_actual
        )
        return pairs, missing, hallucinated
