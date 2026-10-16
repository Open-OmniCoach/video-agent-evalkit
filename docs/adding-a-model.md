# Adding A Model Or Agent

Create one self-contained directory under `examples/`:

```text
examples/<name>/
  __init__.py
  configs/
    run.yaml               required
    serve.yaml             optional local-service topology
    media.yaml             optional integration-owned schema
  generate.py              only for custom model/agent behavior
  README.md
```

Copy [`examples/_template/`](../examples/_template/README.md) and update
`model.generate_fn` to the copied directory's import path. Use the built-in
generator when one chat request is sufficient. Examples are source-only; run
from the repository root with `evalkit run`. The CLI adds the current working directory to the import
path. Config paths remain relative to the YAML file; an absolute config path
does not change where Python modules are imported from.

## Ownership

An integration owns:

- the candidate and auxiliary model endpoints;
- sampling parameters and model-specific request fields;
- system prompts, tools, turn limits, and environment interaction;
- model-specific media sampling, resize, and temporary-file policy;
- normalization of the model or agent's final response into a
  `GenerationResult`;
- optional local serving configuration.

It does not own benchmark annotations, references, rule scorers, judge rubrics,
official aggregators, or reporting facets.

For video and audio preprocessing examples, see
[`examples/qwen3_omni_instruct/`](../examples/qwen3_omni_instruct/README.md).
Media policy is owned by the integration rather than evalkit core.

## Run Configuration

```yaml
registry_path: ../../../configs/registry.yaml
data_root: ../../../data
out_dir: ../../../runs/my-model

scheduler:
  max_active_cases: 64

model:
  endpoint:
    service_manifest: ../../../runs/services/my-model.json
    service_model: candidate
    media_transport: file_uri
    max_inflight: 64

defaults:
  sampling: {temperature: 0, max_tokens: 4096}
  scoring: {type: rule}
  idempotent: true

retry:
  case_max_attempts: 2

eval_datasets:
  - name: omnivideobench
  - name: timelens_bench
```

Paths are relative to the config file. `scheduler.max_active_cases` bounds live
cases; the runtime never creates one task per dataset row.
`endpoint.max_inflight` independently bounds requests to a logical model.
Transport retries cover retryable HTTP failures. Whole-case retries are only
enabled when `idempotent: true`, because replaying an agent may repeat external
side effects. See [serving.md](serving.md#run-endpoint) for the other endpoint
sources.

## Use The Built-In Generator First

For a single OpenAI-compatible chat request, omit `generate_fn`:

```yaml
model:
  endpoint:
    base_url: http://127.0.0.1:8000/v1
    model: my-model
    adapter: vllm_multimodal
    media_transport: file_uri
```

This selects `evalkit.generate.openai_chat:generate`. The benchmark loader has
already produced content-level messages; the transport converts generic image,
video, and audio parts to the configured API adapter.

## Custom Generation Contract

Select any importable function globally or for one benchmark:

```yaml
model:
  generate_fn: examples.my_agent.generate:generate

eval_datasets:
  - name: one_special_bench
    generate_fn: examples.my_agent.special:generate
```

The contract is:

```python
async def generate(runtime, case, sampling) -> GenerationResult:
    ...
```

`case` is an `EvalCase` containing the immutable benchmark identity, messages,
target, problem type, metadata, and reporting facets. `runtime.chat()` is the
only model-service primitive needed by most integrations.

```python
from evalkit.types import GenerationResult

SYSTEM = "You are a tool-using video analyst. Return a concise final answer."


async def generate(runtime, case, sampling):
    messages = ({"role": "system", "content": SYSTEM}, *case.messages)
    chat = await runtime.chat(
        "candidate",
        messages,
        sampling=sampling,
        route_key=case.uid,
    )
    return GenerationResult.completed(
        chat.text,
        assistant_message=chat.message,
        usage=chat.usage,
        timing=chat.timing,
        endpoint=chat.endpoint,
        finish_reason=chat.finish_reason,
    )
```

For agents, retain assistant `tool_calls`, append tool messages with
`tool_call_id`, and use `route_key=case.uid` on every turn. The runtime then keeps
the trajectory on one replica while allowing retryable transport failures to
rebind it. Return trajectory events in `GenerationResult.trajectory`; the driver
writes them to an artifact instead of inflating `predictions.jsonl`.

`evalkit.generate.agent:generate` is a reference loop. A real agent normally
copies its shape into the integration and owns its tool schema and executor.
Start a new integration from [`examples/_template/`](../examples/_template/README.md)
and implement its tool loop in `generate.py`. The template starts with one chat
request; add the prompts, tools and turn budget required by your agent.

## Trajectories

Non-empty trajectories are written to
`<out_dir>/<bench>/artifacts/trajectories/<uid-hash>.json`, and the prediction
row stores that path in `trajectory_artifact`.

Set top-level `trajectory: trainable` to have evalkit capture every model
request and response made during generation as a `model_exchange`; scoring
calls are not captured, and integrations still own tool and environment events.
Within a turn, the exchange follows `case_input`, `media_initial` and
`model_request` events and precedes the turn's other events, so a generator
that makes one candidate call per turn yields one interleaved timeline. Every
event records the zero-based case `attempt` that produced it, and
`runtime.capture_model_io` tells a generator that requests are already
recorded. This guarantees structural replayability, not data quality.

## Full Sample Lifecycle

For a normalized row such as:

```json
{
  "problem_id": "q1",
  "problem_type": "multiple choice",
  "problem": "What sound follows the door closing?",
  "options": ["A. Music", "B. Speech"],
  "solution": "<answer>A</answer>",
  "data_type": "video",
  "path": "./clips/q1.mp4",
  "facets": {"task": "event order"}
}
```

the loader resolves the media path and builds approximately:

```python
EvalCase(
    uid="q1",
    messages=({
        "role": "user",
        "content": [
            {"type": "video", "video": "/abs/data/.../clips/q1.mp4"},
            {"type": "text", "text": "What sound ...\nA. Music\nB. Speech\nAnswer ..."},
        ],
    },),
    target="<answer>A</answer>",
    facets={"task": "event order"},
    ...,
)
```

With `adapter: vllm_multimodal` and `media_transport: file_uri`, the request sent
to `/v1/chat/completions` contains a `video_url` with a `file://` URL plus the
text part. The generate function returns the raw model text. The benchmark
scorer extracts the option letter and produces `ScoreResult(prediction="A",
score=1.0)`. Persistence records both raw and parsed values.

## Retry Boundary

Transport retries are safe for a request that failed before a valid response was
accepted. Whole-case retries replay the generate function and are therefore
opt-in:

```yaml
defaults:
  idempotent: false          # keep false for agents with side effects
retry:
  case_max_attempts: 2
```

Set `idempotent: true` for pure single-turn evaluation. A returned `failed`
generation status is retried only under that policy. Scoring is never retried by
re-running generation.

## Checklist

1. Create a run config and choose direct URLs or a service manifest.
2. Use a built-in generator or copy the integration template for custom behavior.
3. Validate a small benchmark limit and inspect the exact persisted response.
4. Run the smoke config twice to verify resume behavior.
5. Remove limits and use a distinct output directory for the full run.
