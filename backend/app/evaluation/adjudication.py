import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.evaluation.runner import replay_saved_extractions
from app.evaluation.structured_memory import StructuredMemoryEvaluator, normalize
from app.schemas.evaluation import (
    EvaluationAdjudication,
    EvaluationCase,
    EvaluationDataset,
    PostHocAdjudicationReport,
    PostHocMetrics,
)
from app.schemas.memory.extraction import StructuredActivityExtraction


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adjudication(path: str | Path) -> EvaluationAdjudication:
    return EvaluationAdjudication.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def replay_with_adjudication(
    dataset_path: str | Path,
    artifact_path: str | Path,
    adjudication_path: str | Path,
) -> PostHocAdjudicationReport:
    """Replay human-approved aliases without altering the strict source artifacts."""
    dataset_path = Path(dataset_path)
    artifact_path = Path(artifact_path)
    dataset_bytes = dataset_path.read_bytes()
    artifact_bytes = artifact_path.read_bytes()
    adjudication_bytes = Path(adjudication_path).read_bytes()
    dataset = EvaluationDataset.model_validate_json(dataset_bytes)
    artifact: dict[str, Any] = json.loads(artifact_bytes)
    adjudication = EvaluationAdjudication.model_validate_json(adjudication_bytes)

    if adjudication.dataset_name != dataset.name:
        raise ValueError("adjudication dataset name does not match dataset")
    if artifact.get("dataset_name") != dataset.name:
        raise ValueError("extraction artifact dataset name does not match dataset")
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    adjudication_sha256 = hashlib.sha256(adjudication_bytes).hexdigest()
    if adjudication.dataset_sha256 != dataset_sha256:
        raise ValueError("adjudication dataset SHA-256 does not match")
    if adjudication.extraction_artifact_sha256 != artifact_sha256:
        raise ValueError("adjudication extraction artifact SHA-256 does not match")

    actual_by_case = {
        item["case_id"]: StructuredActivityExtraction.model_validate(item["extraction"])
        for item in artifact["actual_extractions"]
    }
    cases_by_id = {case.id: case for case in dataset.cases}
    aliases_by_case = {
        case.id: {key: list(values) for key, values in case.fact_aliases.items()}
        for case in dataset.cases
    }

    for match in adjudication.matches:
        case = cases_by_id.get(match.case_id)
        actual = actual_by_case.get(match.case_id)
        if case is None or actual is None:
            raise ValueError(f"unknown adjudication case: {match.case_id}")
        expected_facts = StructuredMemoryEvaluator._fact_titles(case.expected)[match.kind]
        actual_facts = StructuredMemoryEvaluator._fact_titles(actual)[match.kind]
        expected_title = normalize(match.expected_title)
        actual_title = normalize(match.actual_title)
        if expected_title not in expected_facts:
            raise ValueError(f"unknown expected fact: {match.expected_title}")
        if actual_title not in actual_facts:
            raise ValueError(f"unknown actual fact: {match.actual_title}")
        key = f"{match.kind}:{expected_title}"
        aliases_by_case[match.case_id].setdefault(key, []).append(match.actual_title)

    adjudicated_dataset = dataset.model_copy(
        update={
            "cases": [
                EvaluationCase.model_validate(
                    {
                        **case.model_dump(mode="python"),
                        "fact_aliases": aliases_by_case[case.id],
                    }
                )
                for case in dataset.cases
            ]
        }
    )
    strict_report = replay_saved_extractions(dataset, artifact)
    adjudicated_report = replay_saved_extractions(adjudicated_dataset, artifact)
    return PostHocAdjudicationReport(
        dataset_sha256=dataset_sha256,
        extraction_artifact_sha256=artifact_sha256,
        adjudication_sha256=adjudication_sha256,
        strict_report=strict_report,
        adjudicated_metrics=PostHocMetrics.model_validate(
            adjudicated_report.model_dump(exclude={"passed"})
        ),
    )


def write_post_hoc_report(
    dataset_path: str | Path,
    artifact_path: str | Path,
    adjudication_path: str | Path,
    output_path: str | Path,
) -> PostHocAdjudicationReport:
    """Atomically reserve a new output and remove it if replay or writing fails."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.open("x", encoding="utf-8")
    try:
        with output_file:
            report = replay_with_adjudication(
                dataset_path, artifact_path, adjudication_path
            )
            json.dump(report.model_dump(mode="json"), output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        return report
    except BaseException:
        output_file.close()
        output_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a post-hoc semantic adjudication")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("adjudication", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write_post_hoc_report(
        args.dataset, args.artifact, args.adjudication, args.output
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_type": report.report_type,
                "independent_holdout": report.independent_holdout,
                "strict_passed": report.strict_report.passed,
                "adjudicated_f1": report.adjudicated_metrics.f1,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
