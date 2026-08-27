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

The template is not an evaluation result and must never be reported as real data.
Versioned aggregate run evidence is stored under `results/`; failed runs remain failed and
must not be reclassified by weakening matching rules after observing model output.
