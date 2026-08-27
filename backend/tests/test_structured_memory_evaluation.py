import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.enums.memory import ActivityType, OwnerType, SourceType
from app.evaluation import StructuredMemoryEvaluator, load_dataset
from app.evaluation.runner import evaluate_dataset, report_exit_code, run_evaluation
from app.schemas.evaluation import EvaluationCase, EvaluationDataset, EvaluationThresholds
from app.schemas.memory.extraction import (
    RequirementCandidate,
    ProjectSignal,
    StructuredActivityExtraction,
    TaskCandidate,
)


class MappingExtractor:
    def __init__(self, outputs):
        self.outputs = outputs

    def extract(self, activity):
        return self.outputs[activity.raw_content]


def output(requirement="Support SSO", task="Send POC plan", owner=OwnerType.SELF, review=False):
    return StructuredActivityExtraction(
        summary="Structured summary",
        requirements=[RequirementCandidate(title=requirement, description="SAML required", confidence=0.9)],
        tasks=[TaskCandidate(title=task, owner_type=owner, confidence=0.8)],
        overall_confidence=0.85,
        review_required=review,
    )


def case(case_id, raw_content, expected, *, fact_aliases=None):
    return EvaluationCase(
        id=case_id,
        raw_content=raw_content,
        activity_type=ActivityType.MEETING,
        source_type=SourceType.MANUAL,
        occurred_at=datetime.now(timezone.utc),
        expected=expected,
        fact_aliases=fact_aliases or {},
    )


def test_perfect_dataset_passes_with_exact_metrics():
    expected = output()
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    report = StructuredMemoryEvaluator(MappingExtractor({"record": expected})).evaluate(dataset)
    assert report.passed is True
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.owner_accuracy == 1.0
    assert report.hallucinated_facts == 0


def test_missing_and_hallucinated_facts_fail_gate():
    expected = output()
    actual = output(requirement="Invented requirement", task="Different task")
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual})).evaluate(dataset)
    assert report.passed is False
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.hallucinated_facts == 2
    assert len(report.cases[0].missing_facts) == 2


def test_owner_and_review_required_are_scored():
    expected = output(owner=OwnerType.CUSTOMER, review=True)
    actual = output(owner=OwnerType.SELF, review=False)
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual})).evaluate(dataset)
    assert report.owner_accuracy == 0.0
    assert report.review_required_accuracy == 0.0
    assert report.passed is False


def test_missing_task_cannot_bypass_owner_gate():
    expected = output(owner=OwnerType.CUSTOMER)
    actual = StructuredActivityExtraction(
        summary="summary",
        requirements=expected.requirements,
        overall_confidence=0.8,
    )
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    thresholds = EvaluationThresholds(min_recall=0.5)
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual}), thresholds).evaluate(dataset)
    assert report.owner_accuracy == 0.0
    assert report.passed is False


def test_human_reviewed_alias_matches_deterministically():
    expected = output(task="提供整体规划方案材料", owner=OwnerType.SELF)
    actual = output(task="提供整体规划材料", owner=OwnerType.SELF)
    evaluation_case = case(
        "case-1",
        "record",
        expected,
        fact_aliases={"task:提供整体规划方案材料": ["提供整体规划材料"]},
    )
    dataset = EvaluationDataset(name="test", cases=[evaluation_case])
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual})).evaluate(dataset)
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.owner_accuracy == 1.0
    assert report.hallucinated_facts == 0


def test_unrelated_titles_do_not_match():
    expected = output(requirement="支持终端基线安全")
    actual = output(requirement="提供高可用资源清单")
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual})).evaluate(dataset)
    assert "requirement:支持终端基线安全" in report.cases[0].missing_facts
    assert "requirement:提供高可用资源清单" in report.cases[0].hallucinated_fact_labels


def test_customer_alias_and_negation_are_not_fuzzy_matched():
    expected = output(requirement="支持高可用")
    expected.project_signal = ProjectSignal(customer_name="客户A", confidence=1.0)
    actual = output(requirement="不支持高可用")
    actual.project_signal = ProjectSignal(customer_name="客户B", confidence=1.0)
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual})).evaluate(dataset)
    assert "customer:客户a" in report.cases[0].missing_facts
    assert "customer:客户b" in report.cases[0].hallucinated_fact_labels
    assert "requirement:支持高可用" in report.cases[0].missing_facts
    assert "requirement:不支持高可用" in report.cases[0].hallucinated_fact_labels


def test_invented_project_signal_counts_as_hallucination():
    expected = output()
    actual = output()
    actual.project_signal = ProjectSignal(project_name="Invented project", confidence=0.9)
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual})).evaluate(dataset)
    assert report.hallucinated_facts == 1
    assert report.cases[0].hallucinated_fact_labels == ["project:invented project"]
    assert report.passed is False


def test_duplicate_actual_fact_counts_as_hallucination():
    expected = output()
    actual = output()
    actual.requirements.append(actual.requirements[0].model_copy())
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual})).evaluate(dataset)
    assert report.hallucinated_facts == 1
    assert report.passed is False


def test_thresholds_are_configurable():
    expected = output()
    actual = StructuredActivityExtraction(summary="summary", overall_confidence=0.5)
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    thresholds = EvaluationThresholds(
        min_precision=0,
        min_recall=0,
        min_owner_accuracy=0,
        max_hallucinated_facts=0,
    )
    report = StructuredMemoryEvaluator(MappingExtractor({"record": actual}), thresholds).evaluate(dataset)
    assert report.passed is True


def test_dataset_template_loads():
    path = Path(__file__).parents[1] / "evals" / "dataset.template.json"
    dataset = load_dataset(path)
    assert dataset.name == "replace-with-versioned-deidentified-dataset-name"
    assert len(dataset.cases) == 1


def test_deidentified_real_dataset_loads():
    path = Path(__file__).parents[1] / "evals" / "dataset.real.deidentified.v1.json"
    dataset = load_dataset(path)
    assert dataset.name == "presales-daily-report-deidentified-v1-2026-04"
    assert len(dataset.cases) == 10


def test_versioned_deepseek_result_records_failed_gate():
    path = Path(__file__).parents[1] / "evals" / "results" / "deepseek-chat-v1.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["dataset_name"] == "presales-daily-report-deidentified-v1-2026-04"
    assert result["case_count"] == 10
    assert result["passed"] is False


def test_runner_artifact_omits_raw_content_field():
    expected = output()
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    artifact = evaluate_dataset(
        dataset,
        MappingExtractor({"record": expected}),
        model_name="test:model",
    )
    assert artifact["model"] == "test:model"
    assert artifact["report"]["passed"] is True
    assert artifact["actual_extractions"][0]["case_id"] == "case-1"
    assert artifact["actual_extractions"][0]["extraction"] == expected.model_dump(mode="json")
    assert "raw_content" not in json.dumps(artifact, ensure_ascii=False)


def test_runner_existing_output_fails_before_extractor_creation():
    root = Path(__file__).parents[1]
    existing_output = root / "evals" / "results" / "deepseek-chat-v1.json"

    def forbidden_factory(_model_name):
        raise AssertionError("extractor must not be created")

    with pytest.raises(FileExistsError):
        run_evaluation(
            root / "evals" / "dataset.real.deidentified.v1.json",
            existing_output,
            model_name="test:model",
            extractor_factory=forbidden_factory,
        )


def test_runner_removes_reserved_output_when_extraction_fails():
    root = Path(__file__).parents[1]
    output_path = root / "evals" / "results" / ".runner-failure-test.json"
    output_path.unlink(missing_ok=True)

    class FailingExtractor:
        def extract(self, _activity):
            raise RuntimeError("model failed")

    try:
        with pytest.raises(RuntimeError, match="model failed"):
            run_evaluation(
                root / "evals" / "dataset.real.deidentified.v1.json",
                output_path,
                model_name="test:model",
                extractor_factory=lambda _model_name: FailingExtractor(),
            )
        assert not output_path.exists()
    finally:
        output_path.unlink(missing_ok=True)


def test_runner_exit_code_reflects_gate_result():
    assert report_exit_code({"report": {"passed": True}}) == 0
    assert report_exit_code({"report": {"passed": False}}) == 1


def test_dataset_rejects_blank_records_naive_time_and_empty_cases():
    base = {
        "id": "case-1",
        "raw_content": "record",
        "activity_type": "MEETING",
        "source_type": "MANUAL",
        "occurred_at": "2026-08-27T09:00:00+08:00",
        "expected": output().model_dump(mode="json"),
    }
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**base, "raw_content": "   "})
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate({**base, "occurred_at": "2026-08-27T09:00:00"})
    with pytest.raises(ValidationError):
        EvaluationDataset(name="empty", cases=[])


def test_dataset_rejects_duplicate_case_ids_and_no_expected_facts():
    expected = StructuredActivityExtraction(summary="summary", overall_confidence=1.0)
    empty_case = case("case-1", "record", expected)
    with pytest.raises(ValidationError, match="at least one expected fact"):
        EvaluationDataset(name="empty-ground-truth", cases=[empty_case])

    expected_case = case("case-1", "record", output())
    with pytest.raises(ValidationError, match="case ids must be unique"):
        EvaluationDataset(name="duplicates", cases=[expected_case, expected_case])

    duplicate_fact_case = case("case-2", "record", output())
    duplicate_fact_case.expected.tasks.append(
        TaskCandidate(title="  SEND   poc PLAN ", owner_type=OwnerType.SELF, confidence=0.7)
    )
    with pytest.raises(ValidationError, match="duplicate normalized tasks titles"):
        EvaluationDataset(name="duplicate-facts", cases=[duplicate_fact_case])


def test_case_rejects_unknown_blank_and_ambiguous_aliases():
    expected = output(requirement="支持高可用")
    with pytest.raises(ValidationError, match="unknown labels"):
        EvaluationCase(
            id="case-1",
            raw_content="record",
            activity_type=ActivityType.MEETING,
            occurred_at=datetime.now(timezone.utc),
            expected=expected,
            fact_aliases={"risk:不存在": ["别名"]},
        )
    with pytest.raises(ValidationError, match="must not be blank"):
        EvaluationCase(
            id="case-1",
            raw_content="record",
            activity_type=ActivityType.MEETING,
            occurred_at=datetime.now(timezone.utc),
            expected=expected,
            fact_aliases={"requirement:支持高可用": ["   "]},
        )
    ambiguous_expected = output(requirement="高可用")
    ambiguous_expected.requirements.append(
        RequirementCandidate(title="Support SSO", confidence=1.0)
    )
    with pytest.raises(ValidationError, match="maps to multiple facts"):
        EvaluationCase(
            id="case-1",
            raw_content="record",
            activity_type=ActivityType.MEETING,
            occurred_at=datetime.now(timezone.utc),
            expected=ambiguous_expected,
            fact_aliases={"requirement:高可用": ["Support SSO"]},
        )
