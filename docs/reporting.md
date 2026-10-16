# Reporting Breakdowns

Scoring computes one case's score. Reporting groups already-scored cases by
normalized benchmark taxonomy and applies the same numeric mean within each
group.

## Annotation Facets

Converters normalize upstream fields such as `Type`, `category`, or
`video_duration` under `facets`:

```json
{
  "problem_id": "1",
  "problem_type": "multiple choice",
  "problem": "Which event happened with the sound?",
  "solution": "<answer>B</answer>",
  "facets": {
    "task": "AV Event Alignment",
    "duration": "30s",
    "domain": "Education"
  }
}
```

Raw field names are converter concerns. Aggregation consumes stable facet names.

## Registry Defaults

Official slices belong to the global registry:

```yaml
dailyomni:
  annotation: annotations/dailyomni.json
  video_dir: benchmarks/Daily-Omni/Videos
  status: ready
  report:
    breakdowns:
      - {name: task, key: facets.task}
      - {name: duration, key: facets.duration}
      - name: task_by_duration
        keys: [facets.task, facets.duration]
```

`key` creates a single-facet grouping. `keys` creates an explicit cross-facet
grouping whose labels are joined with ` | `.

## Run Overrides

Ad hoc analysis can replace the registry report for one run:

```yaml
eval_datasets:
  - name: dailyomni
    report:
      breakdowns:
        - {name: task, key: facets.task}
```

Do not move official benchmark slices into each model integration; that makes
cross-model summaries inconsistent.

## Output

Breakdowns are nested under `summary.json -> metrics.breakdowns`:

```json
{
  "metrics": {
    "score": 0.71,
    "n_scored": 1197,
    "breakdowns": {
      "task": {
        "AV Event Alignment": {"score": 0.68, "n_scored": 238},
        "Reasoning": {"score": 0.74, "n_scored": 190}
      }
    }
  }
}
```

Missing facet values are grouped as `__missing__`. Failed, cancelled, and
non-numeric-score rows are excluded from breakdown denominators under the same
rules as the default headline metric.

Write a dataset-specific aggregator only when the metric itself is not a mean of
per-case numeric scores.
