# Scoring Belongs To The Benchmark

Generation identifies the model's final response. Scoring determines whether
that response satisfies the benchmark. Keeping these stages separate guarantees
that all models use the same answer parser and metric.

```text
GenerationResult.response
  -> scorer.extract(response, case.scoring_context())
  -> prediction
  -> scorer.score(prediction, case.scoring_context())
  -> ScoreResult
```

The persisted row keeps `response`, `prediction`, and `score` as distinct fields.

## Rule Scoring

The built-in runtime resolves a scorer in this order:

1. explicit `scoring.scorer` name;
2. scorer registered for the dataset name;
3. the generic `ProblemType` scorer.

```yaml
defaults:
  scoring: {type: rule}

eval_datasets:
  - name: diving48
    scoring: {type: rule, scorer: diving48_f16}
    aggregator: diving48_f16
```

Generic scorers cover multiple choice, numerical, regression, count, yes/no,
temporal grounding, spatial-temporal grounding, and open-ended extraction.

Register a dataset scorer only when per-case correctness cannot be represented by
the generic problem type:

```python
@register_dataset_scorer("my_benchmark")
class MyScorer:
    def extract(self, raw, context): ...
    def score(self, prediction, context): ...
```

## Judge Scoring

An open-ended benchmark can select an auxiliary logical model:

```yaml
aux_models:
  judge:
    endpoint:
      base_url: https://judge.example.com/v1
      api_key_env: JUDGE_API_KEY
      model: judge-model

eval_datasets:
  - name: open_ended_bench
    scoring:
      type: judge
      model: judge
      rubric: |-
        Question: {question}
        Reference: {reference}
        Answer: {answer}
        Return only 1 for correct or 0 for incorrect.
      sampling: {temperature: 0, max_tokens: 8}
```

The rubric is benchmark/run policy; the endpoint is a model resource. Judge
requests use the same persistent transport and appear under the judge model's
runtime metrics. Generation usage and judge usage remain distinguishable.

## Corpus Aggregation

Aggregation runs only after all pending UIDs have a persisted result. Resolution
order is:

1. explicit benchmark `aggregator`;
2. aggregator registered for the dataset name;
3. mean numeric score.

Use a custom aggregator for true corpus protocols such as weighted groups,
ranking/AP, or mIoU plus recall thresholds. Reporting a simple score by category
uses facets instead; see [reporting.md](reporting.md).

Failed and cancelled rows contribute zero to built-in numeric metric denominators
and are counted in the benchmark summary. A completed row whose score is `null`
is reported as unscored rather than silently treated as zero.

## Custom Score Function

For a protocol that needs asynchronous external logic beyond the built-in judge,
select an import path:

```yaml
eval_datasets:
  - name: my_bench
    score_fn: examples.my_model.scoring:score
```

The contract is:

```python
async def score(runtime, case, generation, *, config=None) -> ScoreResult:
    ...
```

The same module can provide a synchronous corpus aggregator:

```yaml
eval_datasets:
  - name: my_bench
    score_fn: examples.my_model.scoring:score
    aggregator: examples.my_model.scoring:aggregate
```

```python
def aggregate(records: list[Prediction]) -> dict:
    ...
```

Scoring failures preserve the generated response, mark the final prediction row
failed, and record `failure_stage: score`. They never trigger another model
generation.
