# Video Agent EvalKit (VAE)

Video Agent EvalKit (VAE) evaluates video agents and multimodal models across
benchmarks with text, image, video and audio questions. Its `evalkit` runner calls
models served through an OpenAI-compatible chat API. It loads
normalized benchmark cases, calls your model or agent, scores each answer with
the benchmark's own scorer, and writes per-case predictions, optional
trajectories and summary metrics. An interrupted run resumes from the cases it
already recorded.

The Python package and command remain `evalkit` for compatibility with existing
configs and integrations.

This is the `omnicoach` integration branch. It includes the trained
[OmniCoach agent](examples/omni_agent/README.md) on top of VAE's shared runner
and template. Use `main` for the general-purpose toolkit.

## What It Is For

Use evalkit to:

- score a checkpoint, a hosted model or an agent on registered benchmarks, with
  the same normalized data and scorers for every model;
- run long evaluations with bounded concurrency across one or more replicas,
  and resume after an interruption;
- keep agent trajectories, optionally including every model request and
  response for training use.

evalkit does not train models, download benchmark data or host results. Model
weights, benchmark annotations and media stay outside the repository; only a
three-question text sample ships in `data/`.

## Requirements

| Requirement | Notes |
| --- | --- |
| Python 3.11 or newer | `.python-version` selects 3.12 for uv. |
| uv 0.11.28, exactly | Every `pyproject.toml` sets `required-version = "==0.11.28"`; other uv versions refuse to sync. |
| A model endpoint | An existing OpenAI-compatible `/v1/chat/completions` service, or NVIDIA GPUs for a local vLLM, SGLang or vLLM-Omni backend. |
| Benchmark files | Obtained from each benchmark's own release; see [docs/data.md](docs/data.md). Not needed for the bundled sample. |

Install the pinned uv version:

```bash
curl -LsSf https://astral.sh/uv/0.11.28/install.sh | sh
# or, for an existing standalone uv installation:
uv self update 0.11.28
uv --version
```

## Install

```bash
git clone https://github.com/Kagura-0001/video-agent-evalkit.git
cd video-agent-evalkit
git switch omnicoach
uv sync --locked
source .venv/bin/activate
```

`uv sync --locked` installs:

- the `evalkit` package and CLI, whose only runtime dependencies are `httpx`
  and `PyYAML`;
- the default `media` dependency group: `static-ffmpeg` (user-space
  FFmpeg/FFprobe), Pillow, and Decord (Decord wheels exist only for Linux
  x86_64 and Windows x86_64). Skip the group with `uv sync --locked --no-group media`
  when the model service reads media paths or URLs itself.

Two extras are optional: `convert` (pandas, pyarrow) for
`evalkit data convert`, and `test` (pytest).

Integrations that process media on the evaluation host need FFmpeg and FFprobe:

```bash
evalkit env sync tools
```

This checks the commands and minimum versions in `environments/tools.toml`.

Local backends have separate locked environments. Install the backend selected
by your example's serve configuration:

```bash
evalkit env sync vllm
# Other configurations can select the sglang or vllm-omni environment.
```

## Quick Start With The Bundled Sample

`configs/registry.yaml` registers `sample_mcq`, three text-only questions in
[`data/sample_mcq.jsonl`](data/sample_mcq.jsonl). Write a run YAML that points
`model.endpoint` at your OpenAI-compatible service:

```yaml
registry_path: configs/registry.yaml
data_root: data

model:
  name: candidate
  generate_fn: evalkit.generate.openai_chat:generate
  endpoint:
    base_url: http://127.0.0.1:8000/v1
    model: my-model
    api_key_env: MODEL_API_KEY

out_dir: runs/sample
scheduler:
  max_active_cases: 32

eval_datasets:
  - name: sample_mcq
```

Then run:

```bash
evalkit run --config my_run.yaml
```

## Results

With `out_dir: runs/sample`:

```text
runs/sample/
  summary.json                  every benchmark's summary plus runtime metrics
  sample_mcq/
    predictions.jsonl           one JSON row per case; also the resume checkpoint
    summary.json                n_total, n_failed, n_unscored, wall_seconds, metrics
    artifacts/trajectories/     only for cases whose generator returned events
```

Each `predictions.jsonl` row records `uid`, `status`, the raw `response`, the
parsed `prediction`, `score`, `score_details`, `error`, `facets`, `generation`
(usage, timing, endpoint and finish reason) and `trajectory_artifact`.

Running the same config again skips every recorded UID, including failed cases.
Use a new `out_dir` for a different model or setting.
[docs/reporting.md](docs/reporting.md) explains per-category breakdowns.

## Evaluate Real Benchmarks

Prepare benchmark data once per data root:

```bash
uv sync --locked --extra convert
evalkit data link --data-root data --manifest configs/media_links.example.yaml
evalkit data convert omnivideobench omnivideo_test_full jointavbench_full \
  video_mme_full video_mme_v2_full futureomni_full \
  lvomnibench timelens_bench worldsense \
  --data-root data --registry-path configs/registry.yaml --validate
evalkit data validate --data-root data --registry-path configs/registry.yaml
```

`data link` symlinks media you already have into `data/` from a manifest
([`configs/media_links.example.yaml`](configs/media_links.example.yaml) shows
the format). Converters read each benchmark's original annotation files and
write `data/annotations/<benchmark>.json` and the matching registry entry.
[docs/data.md](docs/data.md) lists the source layout every converter expects.

Then list the benchmarks under `eval_datasets` in your run config.

## Model And Agent Examples

Start with the [integration template](examples/_template/README.md) to connect
your own model or agent. Run source-only examples from the repository root with
`evalkit run`. Model weights and benchmark data are supplied separately.

| Example | Purpose | Local backend |
| --- | --- | --- |
| [Integration template](examples/_template/README.md) | Custom `generate` function and a three-question smoke run | Existing API or vLLM |
| [Qwen3-Omni Instruct](examples/qwen3_omni_instruct/README.md) | Original Instruct baseline, single-turn native video or timestamped frames, 64K context | vLLM |
| [OmniCoach / Qwen2.5-Omni](examples/omni_agent/configs/qwen2_5_omni/run.yaml) | Trained 7B agent, 96 initial frames, 128K context | vLLM |
| [OmniCoach / Qwen3-Omni](examples/omni_agent/configs/qwen3_omni/run.yaml) | Trained 30B-A3B agent, up to 256 initial frames, 64K context | SGLang |

Instruct owns its media protocols and `QWEN3_OMNI_MEDIA_CACHE_DIR` cache.
The trained agents share `examples.omni_agent.generate:generate`, decoded media
and `OMNI_AGENT_*` environment variables. Follow the
[OmniCoach setup guide](examples/omni_agent/README.md) for checkpoints, model
configs and cache settings.
Examples and their static assets run from the source checkout; the wheel installs
only the `evalkit` library and CLI.

## Use Your Own Model Or Agent

By default `evalkit.generate.openai_chat:generate` sends each case's messages to
the `candidate` model once. For prompts, tools, multi-turn agents or media
preprocessing, point `model.generate_fn` at your own async function:

```yaml
model:
  generate_fn: my_package.my_agent.generate:generate
```

```python
async def generate(runtime, case, sampling) -> GenerationResult:
    ...
```

The generator owns prompts, tools and the final response; scoring stays with
the benchmark. Read [docs/adding-a-model.md](docs/adding-a-model.md), which also
covers run configuration, retries and trajectories.

## Documentation

| To | Read |
| --- | --- |
| Add a model or agent, configure a run, record trajectories | [docs/adding-a-model.md](docs/adding-a-model.md) |
| Prepare benchmark data and the registry | [docs/data.md](docs/data.md) |
| Serve models, route across replicas, configure endpoints | [docs/serving.md](docs/serving.md) |
| Send video, audio or extracted frames | [docs/multimodal.md](docs/multimodal.md) |
| Understand rule scoring and LLM judges | [docs/scoring.md](docs/scoring.md) |
| Break metrics down by category | [docs/reporting.md](docs/reporting.md) |
| Evaluate a subset without copying data | [docs/subsets.md](docs/subsets.md) |

## Repository Layout

```text
configs/registry.yaml       global benchmark and data registry
data/                       normalized annotations and linked raw media
docs/                       guides, reference and design records
evalkit/bench/              loading, rule scoring, aggregation, resume
evalkit/runtime/            bounded scheduler, transport, routing, judge scoring
evalkit/generate/           built-in single-turn and reference agent generation
evalkit/serve/              local backend process supervisor
environments/               locked, isolated backend projects
examples/_template/         starter integration with run and optional serve configs
examples/omni_agent/        OmniCoach agent, model configs and chat templates
examples/qwen3_omni_instruct/ single-turn Instruct baseline and benchmark recipes
tests/                      unit tests, also runnable as scripts
```

`configs/` intentionally has no global run config. A run is model/agent-specific;
the benchmark registry is shared.

## Repository Scope

`main` contains the runner, documentation, a synthetic sample, the integration
template and the Qwen3-Omni Instruct baseline. `omnicoach` extends it with
the trained OmniCoach agent, model configs, chat templates and agent tests.
The repository, including both branches, is private during preparation.
Model checkpoints, benchmark annotations and media,
credentials, service manifests, full prediction artifacts and host-specific
cluster profiles stay outside Git.

## Help And Contributing

Report problems and propose changes through the repository at
<https://github.com/Kagura-0001/video-agent-evalkit>. [CONTRIBUTING.md](CONTRIBUTING.md)
covers development setup, tests and the repository boundary.

## License

MIT; see [LICENSE](LICENSE).
