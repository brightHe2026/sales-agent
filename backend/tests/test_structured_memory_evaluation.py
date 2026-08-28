import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.enums.memory import ActivityType, OwnerType, SourceType
from app.evaluation import StructuredMemoryEvaluator, load_dataset
from app.evaluation.adjudication import (
    replay_with_adjudication,
    sha256_file,
    write_post_hoc_report,
)
from app.evaluation.runner import (
    evaluate_dataset,
    replay_saved_extractions,
    report_exit_code,
    run_evaluation,
)
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

    v2_path = Path(__file__).parents[1] / "evals" / "dataset.real.deidentified.v2.json"
    v2_dataset = load_dataset(v2_path)
    assert v2_dataset.name == "presales-daily-report-deidentified-v2-2026-04"
    assert len(v2_dataset.cases) == 10

    holdout_path = (
        Path(__file__).parents[1]
        / "evals"
        / "dataset.real.deidentified.holdout-v1.json"
    )
    holdout = load_dataset(holdout_path)
    assert holdout.name == "presales-daily-report-independent-holdout-v1"
    assert len(holdout.cases) == 10


def test_versioned_deepseek_result_records_failed_gate():
    path = Path(__file__).parents[1] / "evals" / "results" / "deepseek-chat-v1.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["dataset_name"] == "presales-daily-report-deidentified-v1-2026-04"
    assert result["case_count"] == 10
    assert result["passed"] is False

    replay_path = path.parent / "deepseek-chat-v2-offline-replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["dataset_name"] == "presales-daily-report-deidentified-v2-2026-04"
    assert replay["run_type"] == "offline-replay"
    assert replay["independent_holdout"] is False
    assert replay["passed"] is False

    detailed_path = path.parent / replay["actual_extractions_source"]
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    assert "extraction_prompt_version" not in detailed
    dataset_path = path.parents[1] / "dataset.real.deidentified.v2.json"
    recomputed = replay_saved_extractions(load_dataset(dataset_path), detailed)
    assert recomputed.precision == replay["metrics"]["precision"]
    assert recomputed.recall == replay["metrics"]["recall"]
    assert recomputed.f1 == replay["metrics"]["f1"]
    assert recomputed.owner_accuracy == replay["metrics"]["owner_accuracy"]
    assert recomputed.review_required_accuracy == replay["metrics"]["review_required_accuracy"]
    assert recomputed.hallucinated_facts == replay["metrics"]["hallucinated_facts"]
    assert recomputed.passed is replay["passed"]

    holdout_result_path = path.parent / "deepseek-chat-independent-holdout-v1.json"
    holdout_result = json.loads(holdout_result_path.read_text(encoding="utf-8"))
    holdout_dataset_path = path.parents[1] / "dataset.real.deidentified.holdout-v1.json"
    holdout_report = replay_saved_extractions(
        load_dataset(holdout_dataset_path),
        holdout_result,
    )
    assert holdout_result["model"] == "deepseek:deepseek-chat"
    assert holdout_result["dataset_name"] == "presales-daily-report-independent-holdout-v1"
    assert holdout_result["report"]["dataset_name"] == holdout_result["dataset_name"]
    assert holdout_result["report"]["case_count"] == 10
    assert holdout_report.model_dump(mode="json") == holdout_result["report"]
    assert holdout_report.passed is False


def test_runner_artifact_omits_raw_content_field():
    expected = output()
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    artifact = evaluate_dataset(
        dataset,
        MappingExtractor({"record": expected}),
        model_name="test:model",
    )
    assert artifact["model"] == "test:model"
    assert artifact["extraction_prompt_version"] == "unknown"
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


def test_replay_rejects_saved_outputs_in_wrong_case_order():
    expected = output()
    dataset = EvaluationDataset(
        name="test",
        cases=[case("case-1", "one", expected), case("case-2", "two", expected)],
    )
    extraction = expected.model_dump(mode="json")
    artifact = {
        "actual_extractions": [
            {"case_id": "case-2", "extraction": extraction},
            {"case_id": "case-1", "extraction": extraction},
        ]
    }
    with pytest.raises(ValueError, match="case ids do not match dataset order"):
        replay_saved_extractions(dataset, artifact)


def test_prompt_v2_regression_evidence_recomputes_without_aliases():
    root = Path(__file__).parents[1]
    results = root / "evals" / "results"
    comparison = json.loads(
        (results / "deepseek-chat-dev-prompt-v2-regression.json").read_text(
            encoding="utf-8"
        )
    )
    dataset = load_dataset(root / "evals" / "dataset.real.deidentified.v2.json")
    without_aliases = dataset.model_copy(
        update={
            "cases": [
                evaluation_case.model_copy(update={"fact_aliases": {}})
                for evaluation_case in dataset.cases
            ]
        }
    )
    baseline = json.loads(
        (results / comparison["baseline_artifact"]).read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (results / comparison["candidate_artifact"]).read_text(encoding="utf-8")
    )
    baseline_report = replay_saved_extractions(without_aliases, baseline)
    candidate_report = replay_saved_extractions(without_aliases, candidate)
    assert baseline_report.f1 == comparison["baseline"]["f1"]
    assert candidate_report.f1 == comparison["candidate"]["f1"]
    assert candidate["extraction_prompt_version"] == "structured-memory-v2"
    assert comparison["accepted"] is False
    assert candidate_report.f1 < baseline_report.f1

    refiner_comparison = json.loads(
        (results / "deepseek-chat-dev-refiner-v1-regression.json").read_text(
            encoding="utf-8"
        )
    )
    refiner_artifact = json.loads(
        (results / refiner_comparison["candidate_artifact"]).read_text(encoding="utf-8")
    )
    refiner_report = replay_saved_extractions(without_aliases, refiner_artifact)
    assert refiner_report.f1 == refiner_comparison["candidate"]["f1"]
    assert (
        refiner_artifact["extraction_prompt_version"]
        == "structured-memory-v1+structured-memory-refiner-v1"
    )
    assert refiner_comparison["accepted"] is False
    assert refiner_report.f1 < baseline_report.f1


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


def test_external_adjudication_replays_equivalent_fact_without_mutating_sources():
    expected = output(requirement="兼容较多Win7设备")
    actual = output(requirement="支持大量Win7设备")
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    temp = Path(__file__).parents[1] / "evals" / "results"
    token = uuid4().hex
    dataset_path = temp / f".adjudication-dataset-{token}.json"
    artifact_path = temp / f".adjudication-result-{token}.json"
    adjudication_path = temp / f".adjudication-{token}.json"
    try:
        dataset_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
        artifact = evaluate_dataset(dataset, MappingExtractor({"record": actual}), model_name="test:model")
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        strict_report = replay_saved_extractions(dataset, artifact)
        original_dataset = dataset_path.read_bytes()
        original_artifact = artifact_path.read_bytes()
        adjudication = {
            "dataset_name": "test",
            "dataset_sha256": sha256_file(dataset_path),
            "extraction_artifact_sha256": sha256_file(artifact_path),
            "matches": [{
                "case_id": "case-1", "kind": "requirement",
                "expected_title": "兼容较多Win7设备", "actual_title": "支持大量Win7设备",
            }],
        }
        adjudication_path.write_text(json.dumps(adjudication, ensure_ascii=False), encoding="utf-8")
        adjudicated = replay_with_adjudication(dataset_path, artifact_path, adjudication_path)
        assert strict_report.f1 == 0.5
        assert adjudicated.report_type == "post_hoc_adjudication"
        assert adjudicated.independent_holdout is False
        assert adjudicated.strict_report.f1 == strict_report.f1
        assert adjudicated.adjudicated_metrics.f1 == 1.0
        assert "passed" not in adjudicated.adjudicated_metrics.model_dump()
        assert dataset_path.read_bytes() == original_dataset
        assert artifact_path.read_bytes() == original_artifact
    finally:
        dataset_path.unlink(missing_ok=True)
        artifact_path.unlink(missing_ok=True)
        adjudication_path.unlink(missing_ok=True)


def test_external_adjudication_rejects_tampering_unknown_facts_and_duplicate_mapping():
    expected = output(requirement="Expected")
    actual = output(requirement="Actual")
    dataset = EvaluationDataset(name="test", cases=[case("case-1", "record", expected)])
    temp = Path(__file__).parents[1] / "evals" / "results"
    token = uuid4().hex
    dataset_path = temp / f".adjudication-invalid-dataset-{token}.json"
    artifact_path = temp / f".adjudication-invalid-result-{token}.json"
    adjudication_path = temp / f".adjudication-invalid-{token}.json"
    try:
        dataset_path.write_text(dataset.model_dump_json(), encoding="utf-8")
        artifact = evaluate_dataset(dataset, MappingExtractor({"record": actual}), model_name="test:model")
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        base = {
            "dataset_name": "test",
            "dataset_sha256": sha256_file(dataset_path),
            "extraction_artifact_sha256": sha256_file(artifact_path),
            "matches": [{"case_id": "case-1", "kind": "requirement", "expected_title": "Missing", "actual_title": "Actual"}],
        }
        adjudication_path.write_text(json.dumps(base), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown expected fact"):
            replay_with_adjudication(dataset_path, artifact_path, adjudication_path)
        base["matches"][0]["expected_title"] = "Expected"
        base["matches"].append(dict(base["matches"][0]))
        adjudication_path.write_text(json.dumps(base), encoding="utf-8")
        with pytest.raises(ValidationError, match="adjudicated only once"):
            replay_with_adjudication(dataset_path, artifact_path, adjudication_path)
        base["matches"] = base["matches"][:1]
        base["dataset_sha256"] = "0" * 64
        adjudication_path.write_text(json.dumps(base), encoding="utf-8")
        with pytest.raises(ValueError, match="dataset SHA-256"):
            replay_with_adjudication(dataset_path, artifact_path, adjudication_path)
        base["dataset_sha256"] = sha256_file(dataset_path)
        artifact["dataset_name"] = "different-dataset"
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        base["extraction_artifact_sha256"] = sha256_file(artifact_path)
        adjudication_path.write_text(json.dumps(base), encoding="utf-8")
        with pytest.raises(ValueError, match="artifact dataset name"):
            replay_with_adjudication(dataset_path, artifact_path, adjudication_path)
    finally:
        dataset_path.unlink(missing_ok=True)
        artifact_path.unlink(missing_ok=True)
        adjudication_path.unlink(missing_ok=True)


def test_approved_holdout_adjudication_writes_explicit_post_hoc_report():
    root = Path(__file__).parents[1] / "evals"
    output_path = root / "results" / f".post-hoc-{uuid4().hex}.json"
    try:
        report = write_post_hoc_report(
            root / "dataset.real.deidentified.holdout-v1.json",
            root / "results" / "deepseek-chat-independent-holdout-v1.json",
            root / "reviews" / "holdout-v1.adjudication.json",
            output_path,
        )
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert report.report_type == "post_hoc_adjudication"
        assert report.independent_holdout is False
        assert report.strict_report.passed is False
        assert report.adjudicated_metrics.f1 == pytest.approx(0.5)
        assert saved == report.model_dump(mode="json")
        assert "passed" not in saved["adjudicated_metrics"]
    finally:
        output_path.unlink(missing_ok=True)


def test_post_hoc_writer_never_overwrites_and_cleans_failed_reservation():
    root = Path(__file__).parents[1] / "evals"
    existing = root / "results" / f".post-hoc-existing-{uuid4().hex}.json"
    failed = root / "results" / f".post-hoc-failed-{uuid4().hex}.json"
    missing_adjudication = root / "reviews" / f".missing-{uuid4().hex}.json"
    dataset = root / "dataset.real.deidentified.holdout-v1.json"
    artifact = root / "results" / "deepseek-chat-independent-holdout-v1.json"
    existing.write_text("sentinel", encoding="utf-8")
    try:
        with pytest.raises(FileExistsError):
            write_post_hoc_report(
                dataset,
                artifact,
                root / "reviews" / "holdout-v1.adjudication.json",
                existing,
            )
        assert existing.read_text(encoding="utf-8") == "sentinel"

        with pytest.raises(FileNotFoundError):
            write_post_hoc_report(dataset, artifact, missing_adjudication, failed)
        assert not failed.exists()
    finally:
        existing.unlink(missing_ok=True)
        failed.unlink(missing_ok=True)
