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
