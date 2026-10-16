# Qwen3-Omni-30B-A3B-Instruct

This maintained integration evaluates the text-output thinker through vLLM's
OpenAI-compatible API. Benchmark annotations, labels, scoring, and reporting
slices remain shared under `configs/registry.yaml` and `evalkit/bench/`.

## Prepare

Run from the repository root. This source-only example evaluates the original
Instruct model in one turn.

```bash
uv sync --locked
source .venv/bin/activate
evalkit env sync vllm
evalkit env check tools
evalkit data validate --data-root data --registry-path configs/registry.yaml
```

Download the checkpoint to a user-owned model directory, or let the serving
backend resolve the Hugging Face model ID:

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir ~/models/Qwen3-Omni-30B-A3B-Instruct
```

## Serve

Copy the supplied topology and set your checkpoint path and GPU settings:

```bash
cp examples/qwen3_omni_instruct/configs/serve.yaml examples/qwen3_omni_instruct/configs/serve.local.yaml
export QWEN3_OMNI_MEDIA_CACHE_DIR=/tmp/evalkit-media/qwen3-omni-64k
evalkit serve --config examples/qwen3_omni_instruct/configs/serve.local.yaml --dry-run
evalkit serve --config examples/qwen3_omni_instruct/configs/serve.local.yaml
```

The example uses two GPUs with tensor parallelism 2 and a 65,536-token service
limit. Adjust TP, GPU IDs, replica count, ports and memory utilization for your machine.
The service writes `runs/services/qwen3-omni-instruct.json` and exposes
`qwen3-omni-instruct`. Its allowed media path comes from the cache environment
variable; the launcher resolves symlinks and creates the directory before startup.

## Canonical Runs

The checked-in runs use local `file://` media. In a second shell, activate the
core environment and export the same absolute cache path as the service before
smoke or full evaluation. Both processes must see the same files:

```bash
export QWEN3_OMNI_MEDIA_CACHE_DIR=/tmp/evalkit-media/qwen3-omni-64k
```

```bash
evalkit run --config examples/qwen3_omni_instruct/configs/run/smoke.yaml

evalkit run --config examples/qwen3_omni_instruct/configs/run/dailyomni.yaml
evalkit run --config examples/qwen3_omni_instruct/configs/run/omnivideobench.yaml
evalkit run --config examples/qwen3_omni_instruct/configs/run/lvomnibench.yaml
evalkit run --config examples/qwen3_omni_instruct/configs/run/worldsense.yaml
evalkit run --config examples/qwen3_omni_instruct/configs/run/timelens.yaml
```

Daily-Omni uses native video plus the annotation's separate audio input. vLLM
samples at 1 FPS, caps at 256 frames, and limits each frame to `384 * 384`
pixels.

OmniVideoBench and LVOmniBench use `long_video.py`. Videos above the 256-frame
budget are normalized once into a uniformly sampled `.mkv` proxy with factor-32 frame
dimensions and 16 kHz mono Vorbis audio. The cache is opt-in, may be deleted
after the run, and is not managed by EvalKit
core. Audio longer than the configured media budget is truncated explicitly and
recorded in the trajectory. Backend media errors fail the case; the integration
does not silently remove audio and retry with a different protocol.

WorldSense uses timestamped JPEG images at 1 FPS, at most 256 uniformly spaced
frames, and the source video's audio as a separate Qwen audio input.

TimeLens uses native video with `use_audio_in_video: true`, 1 FPS and a 256-frame
cap, as specified in its run config.

The maintained run configs assume a 65,536-token service limit. Frame and media
budgets target that bound; the integration does not claim to token-truncate an
arbitrary overlength multimodal request.

Predictions, summaries, timing, usage, and media-policy trajectory events are
written under each run configuration's `out_dir`.
