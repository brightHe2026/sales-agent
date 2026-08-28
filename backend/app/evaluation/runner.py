import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic_ai.models import infer_model

from app.evaluation.structured_memory import StructuredMemoryEvaluator, load_dataset
from app.extraction import (
    ActivityExtractor,
    GroundedActivityExtractor,
    PydanticActivityExtractor,
    PydanticEvidenceGrounder,
)
from app.models import Activity
from app.schemas.evaluation import EvaluationDataset, EvaluationReport
from app.schemas.memory.extraction import StructuredActivityExtraction


class RecordingExtractor:
    def __init__(self, extractor: ActivityExtractor) -> None:
        self.extractor = extractor
        self.outputs: list[StructuredActivityExtraction] = []

    def extract(self, activity: Activity) -> StructuredActivityExtraction:
        output = self.extractor.extract(activity)
        self.outputs.append(output)
        return output


class ReplayExtractor:
    def __init__(self, outputs: list[StructuredActivityExtraction]) -> None:
        self.outputs = iter(outputs)

    def extract(self, _activity: Activity) -> StructuredActivityExtraction:
        return next(self.outputs)


def evaluate_dataset(
    dataset: EvaluationDataset,
    extractor: ActivityExtractor,
    *,
    model_name: str,
) -> dict[str, Any]:
    recording = RecordingExtractor(extractor)
    report = StructuredMemoryEvaluator(recording).evaluate(dataset)
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "extraction_prompt_version": getattr(extractor, "prompt_version", "unknown"),
        "dataset_name": dataset.name,
        "report": report.model_dump(mode="json"),
        "actual_extractions": [
            {
                "case_id": case.id,
                "extraction": output.model_dump(mode="json"),
            }
            for case, output in zip(dataset.cases, recording.outputs, strict=True)
        ],
    }


def replay_saved_extractions(
    dataset: EvaluationDataset,
    artifact: dict[str, Any],
) -> EvaluationReport:
    saved = artifact["actual_extractions"]
    case_ids = [item["case_id"] for item in saved]
    expected_case_ids = [case.id for case in dataset.cases]
    if case_ids != expected_case_ids:
        raise ValueError("saved extraction case ids do not match dataset order")
    outputs = [
        StructuredActivityExtraction.model_validate(item["extraction"])
        for item in saved
    ]
    return StructuredMemoryEvaluator(ReplayExtractor(outputs)).evaluate(dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run structured-memory evaluation")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", default="deepseek:deepseek-chat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ground-evidence",
        action="store_true",
        help="run a second, fact-immutable pass that attaches exact source quotes",
    )
    return parser


def build_extractor(model_name: str) -> ActivityExtractor:
    return PydanticActivityExtractor(infer_model(model_name))


def build_grounded_extractor(model_name: str) -> ActivityExtractor:
    model = infer_model(model_name)
    return GroundedActivityExtractor(
        PydanticActivityExtractor(model), PydanticEvidenceGrounder(model)
    )


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    *,
    model_name: str,
    extractor_factory: Callable[[str], ActivityExtractor] = build_extractor,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.open("x", encoding="utf-8")
    try:
        with output_file:
            dataset = load_dataset(dataset_path)
            extractor = extractor_factory(model_name)
            artifact = evaluate_dataset(dataset, extractor, model_name=model_name)
            json.dump(artifact, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        return artifact
    except BaseException:
        output_file.close()
        output_path.unlink(missing_ok=True)
        raise


def report_exit_code(artifact: dict[str, Any]) -> int:
    return 0 if artifact["report"]["passed"] else 1


def main() -> int:
    args = build_parser().parse_args()
    artifact = run_evaluation(
        args.dataset,
        args.output,
        model_name=args.model,
        extractor_factory=(
            build_grounded_extractor if args.ground_evidence else build_extractor
        ),
    )
    report = artifact["report"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": report["passed"],
                "precision": report["precision"],
                "recall": report["recall"],
                "f1": report["f1"],
            },
            ensure_ascii=False,
        )
    )
    return report_exit_code(artifact)


if __name__ == "__main__":
    raise SystemExit(main())
