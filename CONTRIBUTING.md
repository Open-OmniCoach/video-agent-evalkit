# Contributing

Video Agent EvalKit (VAE) is developed at
<https://github.com/Kagura-0001/video-agent-evalkit>. The `main`
branch is the portable evaluation core: contributions should work without
access to a particular shared filesystem, model checkpoint, or GPU cluster.
The `omnicoach` branch adds the OmniCoach integration. Develop shared
runner changes on `main` and merge them into the integration branch.

## Development setup

Install uv 0.11.28 exactly; every `pyproject.toml` pins it with
`required-version`, and other uv versions refuse to sync. See
[README.md](README.md#requirements).

```bash
uv sync --locked --extra test --extra convert
source .venv/bin/activate
pytest
```

`uv sync` keeps only what one command requests, so pass every extra you need
together. The default `media` group (static-ffmpeg, Pillow, Decord) is included
unless you add `--no-group media`. The local media test skips itself when
FFmpeg or FFprobe is unavailable; to exercise it, provide the host tools first:

```bash
evalkit env sync tools
```

Every test module is also a directly executable script, for example
`python tests/test_runtime_transport.py`.

Backend environments remain isolated from the core package:

```bash
evalkit env sync vllm
evalkit env sync vllm-omni
evalkit env sync sglang
```

## Repository boundary

Before committing:

- replace private model and dataset paths with placeholders or user-supplied
  configuration;
- exclude credentials, cookies, service manifests, full predictions, and raw
  media;
- keep host-specific replica layouts and cluster compatibility settings in
  local configuration unless they are broadly reproducible;
- include a small synthetic fixture or unit test for new behavior;
- run `git diff --check` and the tests relevant to the change.

Compact benchmark summaries may be proposed when their data source, model,
configuration, and reproducibility boundary are documented. Large generated
artifacts belong outside Git.

## Change design

Keep benchmark loading, generation policy, inference transport, and scoring as
separate contracts. A model integration may customize prompts and media
handling, but should not silently change benchmark labels or scoring semantics.
