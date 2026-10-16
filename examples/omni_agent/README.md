# OmniCoach video-QA agent

Qwen2.5-Omni and Qwen3-Omni use one entry point,
`examples.omni_agent.generate:generate`. Model-specific run/serve settings live
in `configs/`; their matching chat templates live in `templates/`.
Supply a checkpoint trained to emit `<observation>`, `<think>` and native tool
calls or `<answer>`, plus the benchmark data. The base/Instruct checkpoints
do not provide this trained agent behavior.

| Recipe | Backend | Context | Initial frames | Default benchmarks |
| --- | --- | --- | --- | --- |
| `qwen2_5_omni` | vLLM | 131072 | 96 | LVOmniBench |
| `qwen3_omni` | SGLang | 65536 | 256 | DailyOmni, OmniVideoBench, LVOmniBench, WorldSense |

Both recipes allow 32 turns and 4096 output tokens per turn. Tools use absolute
seconds, including high-resolution frames and ordered multi-span evidence.
Only the latest media bundle is retained in the conversation. Initial audio
covers videos up to 3300 seconds; longer audio remains available through tools.
Qwen2.5 uses one case attempt and no transport retries. Qwen3 allows two case
attempts and three transport retries. Truncated responses remain terminal.

## Prepare

Use the `codex/omnicoach` branch and run from the repository root on a Linux host:

```bash
uv sync --locked
source .venv/bin/activate
evalkit env sync tools
evalkit env sync vllm
```

For Qwen3, sync `sglang` instead of `vllm`. Backend environments remain isolated
from the evaluation runner and from each other. Prepare benchmarks following
the [data guide](../../docs/data.md).

These examples run from a source checkout. The wheel installs the `evalkit`
library and CLI; it does not install the top-level `examples` package. The
`evalkit run` loads source-only generators from the current working directory.

## Configure and run

Select `qwen2_5_omni` or `qwen3_omni` and create local copies in the same directory:

```bash
recipe=qwen2_5_omni
cp "examples/omni_agent/configs/$recipe/run.yaml" "examples/omni_agent/configs/$recipe/run.local.yaml"
cp "examples/omni_agent/configs/$recipe/serve.yaml" "examples/omni_agent/configs/$recipe/serve.local.yaml"
export OMNI_AGENT_MEDIA_CACHE_ROOT=/tmp/evalkit-media/omni-agent
```

In `serve.local.yaml`, set your checkpoint path, GPU IDs and tensor parallelism.
The supplied topology is one GPU for Qwen2.5 and two GPUs for Qwen3; adjust it
to your machine. Keep the matching chat template. The server and runner must
see the same cache files. The vLLM recipe reads `--allowed-local-media-path`
from `OMNI_AGENT_MEDIA_CACHE_ROOT`; export the same absolute path in both shells.
The launcher resolves symlinks and creates the directory before starting vLLM.
An unset cache variable is reported before launch.

Qwen2.5's serve config applies YaRN factor 4 and a 131072-token context through
`--hf-overrides` at `thinker_config.text_config`. It uses the original checkpoint
directory directly and preserves its other mRoPE parameters.

In `run.local.yaml`, set `data_root` and `registry_path` if using another data
location. Paths resolve relative to the YAML file. Set `EVALKIT_API_KEY` only
when your endpoint requires authentication.

```bash
evalkit serve --config "examples/omni_agent/configs/$recipe/serve.local.yaml" --dry-run
evalkit serve --config "examples/omni_agent/configs/$recipe/serve.local.yaml"
```

In a second shell, activate the core environment, set the same `recipe` and
cache variable, then run:

```bash
evalkit run --config "examples/omni_agent/configs/$recipe/run.local.yaml"
```

The run and serve configs share a model-specific manifest under `runs/services/`.
Use a new prediction output directory when changing a checkpoint or media policy.

## Shared media cache

Both models use `OMNI_AGENT_MEDIA_CACHE_ROOT`, defaulting to
`<system temporary directory>/evalkit-media/omni-agent` (`/tmp/...` on Linux).
Within it, benchmark, `media_cache_id` and media identity select the cache.
Source bytes and preprocessing settings are checked before cached media is reused.

The immutable initial cache contains up to 256 frames sampled at 0.5 FPS.
`defaults.budget.initial_frames` selects 1–256 frames uniformly from that cache,
preserving their timestamps; it defaults to 256. Qwen2.5 sets it to 96.
This view limit does not change the cache contract or trigger another decode.
Turn, retry, context, output-token and initial-audio budgets also require positive
integers; strings, booleans and fractional values are rejected.

`OMNI_AGENT_TRACE_COMPLETE` defaults to `1`; set it to `0` to omit media parts
from the agent's media trace events. `OMNI_AGENT_PRESERVE_TRACE_MEDIA=1` retains
temporary media views referenced by trajectories. By default those views are
removed after each case; the shared decoded cache remains.

Offline tests cover configuration paths, launch arguments, frame selection,
cache sharing and tool turns. GPU serving and benchmark accuracy require the
corresponding checkpoint, data and hardware.
