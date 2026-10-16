# Integration Template

Copy this directory to `examples/<model_or_agent>/`.
Update `model.generate_fn` in the copied run config to
`examples.<model_or_agent>.generate:generate` and choose a distinct `out_dir`.

- `configs/run.yaml` selects benchmarks, generation policy, endpoint, sampling,
  and output.
- `configs/serve.yaml` optionally starts local vLLM, vLLM-Omni, or SGLang
  replicas and writes the service manifest consumed by `configs/run.yaml`.
- `generate.py` is only needed for custom model or agent behavior. Delete it and
  omit `model.generate_fn` to use the built-in single-turn chat generator.
The generate contract is:

```python
async def generate(runtime, case, sampling) -> GenerationResult:
    ...
```

The integration may own system prompts, tools, and turn budgets. It must not own
benchmark labels, scoring, or official reporting slices. Those remain in
`configs/registry.yaml` and `evalkit/bench/`.

To use an existing OpenAI-compatible endpoint, copy `configs/run.yaml` to
`configs/run.local.yaml` in the same directory, set `endpoint.base_url` and
`endpoint.model`, then run from the repository root:

```bash
evalkit run --config examples/_template/configs/run.local.yaml
```

For local serving, set the checkpoint and GPU configuration in `serve.yaml`,
replace `base_url` and `model` in the run endpoint with
`service_manifest: ../../../runs/services/template.json` and
`service_model: candidate`, then run:

```bash
evalkit env sync vllm
evalkit serve --config examples/_template/configs/serve.yaml
# In another shell:
evalkit run --config examples/_template/configs/run.yaml
```

Run the same config again to check that the recorded cases are skipped.
See [../../docs/adding-a-model.md](../../docs/adding-a-model.md) and
[../../docs/serving.md](../../docs/serving.md). For video and audio processing,
see [../qwen3_omni_instruct/README.md](../qwen3_omni_instruct/README.md).
