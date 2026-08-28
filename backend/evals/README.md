# POC-01 Structured Memory Evaluation

Copy `dataset.template.json` to a new versioned JSON file and replace the placeholder
with 5–10 real, de-identified presales records. Do not add customer secrets, quotations,
contracts, credentials, or identifying personal information.

Each expected extraction is human-reviewed ground truth. Confidence values are required
by the runtime schema but are not scored. Project/customer signals and candidate facts use
normalized exact matching. A case may explicitly list human-reviewed `fact_aliases` for
equivalent wording; unlisted paraphrases never match. Task owner type/name and the
review-required decision are scored separately. A missing expected task counts as a failed
owner check.

Alias keys use `<kind>:<normalized canonical title>`, for example:
`"task:提供整体规划方案材料": ["提供整体规划材料"]`. An alias must preserve the
same fact, polarity, subject, and task responsibility; it must never be used to equate
different customers, projects, owners, or positive/negative statements.

Default PASS thresholds:

- precision >= 0.90;
- recall >= 0.85;
- task-owner accuracy >= 0.90;
- review-required accuracy = 1.00;
- zero hallucinated facts.

In reports, `hallucinated_facts` is the historical field name for actual output labels that
did not match an expected canonical title or approved alias. It is a conservative Gate
signal, not proof that every unmatched label was fabricated; human review must distinguish
unsupported facts from paraphrases, granularity differences, and classification drift.

The template is not an evaluation result and must never be reported as real data.
Versioned aggregate run evidence is stored under `results/`; failed runs remain failed and
must not be reclassified by weakening matching rules after observing model output.

Post-run semantic review must be stored in a separate adjudication file. It binds to the
exact dataset and extraction artifact with SHA-256 hashes and may only add human-approved,
one-to-one equivalences between facts of the same kind. Replaying an adjudication produces
a separate post-hoc report: it never replaces the original strict Gate, never makes the
holdout independent again, and never changes either source file. This distinction lets us
measure paraphrase effects without presenting an observed-and-adjudicated result as a new
independent acceptance test. The post-hoc envelope is explicitly marked
`report_type: post_hoc_adjudication` and `independent_holdout: false`; its adjudicated
metrics deliberately have no `passed` field.

Generate a separately marked post-hoc report with an approved adjudication:

```powershell
python -m app.evaluation.adjudication evals/dataset.json evals/results/model-run.json `
  evals/reviews/approved-adjudication.json --output evals/results/post-hoc.json
```

The output path is reserved before replay and must not already exist.

## Exact source evidence

New derived facts carry `source_quote`, an exact substring of the source activity. The
application rejects an entire memory update when any requirement, task, decision, or risk
has no quote or a quote that is not present verbatim in `raw_content`. Quotes are excluded
from semantic fingerprints, so selecting a different supporting span cannot create a
duplicate fact. Database columns remain nullable only for backward compatibility with
facts created before migration `0003_fact_source_quotes`.

Evidence grounding is intentionally a second stage. The first extraction stays immutable;
the grounder may only attach quotes by list position. For development diagnostics, partial
mode records exact quotes and leaves ungrounded facts empty. Production validation remains
strict and blocks persistence when any quote is empty.

Ground a frozen development artifact without re-extracting its facts:

```powershell
python -m app.evaluation.grounding evals/dataset.json evals/results/extractions.json `
  --model deepseek:deepseek-chat --output evals/results/grounded.json
```

A grounding Gate passes only with 100% quote coverage and 100% exact-quote accuracy.
This proves traceability to verbatim source text, not semantic entailment: an exact quote
can still be irrelevant, incomplete, or attached to the wrong fact. Human adjudication or
a separately validated entailment check remains necessary before claiming semantic support.

Run the DeepSeek evaluation from `backend/` with `DEEPSEEK_API_KEY` configured:

```powershell
python -m app.evaluation.runner evals/dataset.real.deidentified.v1.json `
  --model deepseek:deepseek-chat `
  --output evals/results/deepseek-chat-run.json
```

The output path must not already exist. The artifact includes per-case model extractions
for human alias adjudication and an aggregate report; it never stores the API key. The
Runner reserves the output before any model call. If a call or write fails, it removes the
incomplete artifact; retrying will call the model again for every case.
Model-generated summaries, titles, and descriptions may quote or paraphrase source content;
treat every result artifact with the same sensitivity as its evaluation dataset.
