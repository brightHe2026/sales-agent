import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.models import infer_model

from app.enums.memory import ExtractionStatus
from app.extraction.evidence import EvidenceGrounder, PydanticEvidenceGrounder
from app.models import Activity
from app.schemas.evaluation import (
    EvaluationDataset,
    GroundingCaseResult,
    GroundingReport,
)
from app.schemas.memory.extraction import StructuredActivityExtraction


def evaluate_saved_grounding(
    dataset: EvaluationDataset, artifact: dict[str, Any]
) -> GroundingReport:
    saved = artifact["actual_extractions"]
    case_ids = [item["case_id"] for item in saved]
    expected_case_ids = [case.id for case in dataset.cases]
    if case_ids != expected_case_ids:
        raise ValueError("saved extraction case ids do not match dataset order")

    cases: list[GroundingCaseResult] = []
    for case, item in zip(dataset.cases, saved, strict=True):
        extraction = StructuredActivityExtraction.model_validate(item["extraction"])
        groups = {
            "requirement": extraction.requirements,
            "task": extraction.tasks,
            "decision": extraction.decisions,
            "risk": extraction.risks,
        }
        missing: list[str] = []
        invalid: list[str] = []
        fact_count = 0
        quoted_count = 0
        exact_count = 0
        for kind, candidates in groups.items():
            for candidate in candidates:
                fact_count += 1
                label = f"{kind}:{candidate.title}"
                if candidate.source_quote is None:
                    missing.append(label)
                    continue
                quoted_count += 1
                if candidate.source_quote in case.raw_content:
                    exact_count += 1
                else:
                    invalid.append(label)
        cases.append(
            GroundingCaseResult(
                case_id=case.id,
                fact_count=fact_count,
                quoted_fact_count=quoted_count,
                exact_quote_matches=exact_count,
                missing_quote_labels=missing,
                invalid_quote_labels=invalid,
            )
        )

    fact_count = sum(case.fact_count for case in cases)
    quoted_count = sum(case.quoted_fact_count for case in cases)
    exact_count = sum(case.exact_quote_matches for case in cases)
    quote_coverage = quoted_count / fact_count if fact_count else 1.0
    exact_quote_accuracy = exact_count / quoted_count if quoted_count else float(fact_count == 0)
    return GroundingReport(
        dataset_name=dataset.name,
        case_count=len(cases),
        fact_count=fact_count,
        quote_coverage=quote_coverage,
        exact_quote_accuracy=exact_quote_accuracy,
        passed=quote_coverage == 1.0 and exact_quote_accuracy == 1.0,
        cases=cases,
    )


def ground_saved_extractions(
    dataset: EvaluationDataset,
    artifact: dict[str, Any],
    grounder: EvidenceGrounder,
) -> dict[str, Any]:
    saved = artifact["actual_extractions"]
    case_ids = [item["case_id"] for item in saved]
    expected_case_ids = [case.id for case in dataset.cases]
    if case_ids != expected_case_ids:
        raise ValueError("saved extraction case ids do not match dataset order")

    grounded_outputs = []
    for case, item in zip(dataset.cases, saved, strict=True):
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
        extraction = StructuredActivityExtraction.model_validate(item["extraction"])
        grounded = grounder.ground(activity, extraction)
        grounded_outputs.append(
            {"case_id": case.id, "extraction": grounded.model_dump(mode="json")}
        )

    result = {
        **artifact,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "extraction_prompt_version": (
            f"{artifact.get('extraction_prompt_version', 'structured-memory-v1-legacy-artifact')}+"
            f"{getattr(grounder, 'version', 'unknown')}"
        ),
        "run_type": "development-frozen-fact-evidence-grounding",
        "independent_holdout": False,
        "source_extraction_artifact": None,
        "actual_extractions": grounded_outputs,
    }
    result["grounding_report"] = evaluate_saved_grounding(dataset, result).model_dump(
        mode="json"
    )
    return result


def run_grounding(
    dataset_path: str | Path,
    artifact_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
) -> dict[str, Any]:
    from app.evaluation.structured_memory import load_dataset

    dataset_path = Path(dataset_path)
    artifact_path = Path(artifact_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.open("x", encoding="utf-8")
    try:
        with output_file:
            dataset = load_dataset(dataset_path)
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            grounded = ground_saved_extractions(
                dataset,
                artifact,
                PydanticEvidenceGrounder(
                    infer_model(model_name), allow_partial=True
                ),
            )
            grounded["model"] = model_name
            grounded["source_extraction_artifact"] = artifact_path.name
            json.dump(grounded, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        return grounded
    except BaseException:
        output_file.close()
        output_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach exact evidence to a frozen extraction artifact"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", default="deepseek:deepseek-chat")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_grounding(
        args.dataset, args.artifact, args.output, model_name=args.model
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                **{
                    key: result["grounding_report"][key]
                    for key in (
                        "fact_count",
                        "quote_coverage",
                        "exact_quote_accuracy",
                        "passed",
                    )
                },
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["grounding_report"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
