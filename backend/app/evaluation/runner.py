import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic_ai.models import infer_model

from app.evaluation.structured_memory import StructuredMemoryEvaluator, load_dataset
from app.extraction import ActivityExtractor, PydanticActivityExtractor
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
    return parser


def build_extractor(model_name: str) -> ActivityExtractor:
    return PydanticActivityExtractor(infer_model(model_name))


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
