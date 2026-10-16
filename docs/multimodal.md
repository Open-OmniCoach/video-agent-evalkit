# Multimodal Transport

Benchmark annotations describe media; serving backends decode it. The framework
does not install model processors, video decoders, CUDA libraries, or model code
in the accuracy runner.

## Normalized Message Parts

Loaders emit model-independent content parts:

```python
{"type": "image", "image": "/absolute/path/image.jpg"}
{"type": "video", "video": "/absolute/path/video.mp4"}
{"type": "audio", "audio": "/absolute/path/audio.wav"}
{"type": "text", "text": "Question ..."}
```

The configured transport adapter converts these immediately before the HTTP
request. Existing OpenAI-style parts pass through unchanged.

## Media Transport

`endpoint.media_transport` controls local paths:

- `inline` reads a local file and sends a base64 data URI. Use it when the server
  cannot access the evaluation filesystem.
- `file_uri` sends an absolute `file://` URI. Use it for colocated local servers;
  it avoids base64 expansion and repeated client-side file reads.

HTTP(S), data, and file URIs already present in annotations pass through.

Local vLLM/vLLM-Omni must be started with an allowed local-media root. The
example serve configs use `--allowed-local-media-path data` from a deterministic
working directory. Grant only the prepared data root, not the whole filesystem.

## Adapters

`openai` uses standard image parts and inline `input_audio` for local audio.
Remote audio URLs pass through as `audio_url`.

`vllm_multimodal` emits `image_url`, `video_url`, and `audio_url` parts. It is the
default choice for local vLLM, vLLM-Omni, and SGLang multimodal APIs that accept
those extensions.

Adapters preserve assistant tool calls, tool-call IDs, names, and all other
message fields. Only media content is normalized.

## Processing Policy Belongs To The Integration

Frame rate, maximum frames, resize limits, audio extraction, and tool behavior
change what the model sees. They are model or agent policy, not benchmark
conversion or scoring policy.

For a model that accepts raw video, configure its official processor through
backend flags and send the original local file. For an image-only model or
dynamic agent, a custom generate function may explicitly replace the video part
with image parts before calling `runtime.chat()`. Evalkit core intentionally has
no implicit preprocess hook, and the package does not depend on a video
decoder; Decord and Pillow come only from the checkout's default `media` uv
group, which `uv sync --locked --no-group media` skips.

The [Qwen3-Omni Instruct example](../examples/qwen3_omni_instruct/README.md)
provides native-video and timestamped-frame policies, with explicit frame,
resolution and audio budgets. Its cache is owned by the integration. Custom
agents can implement media inspection tools in their own generate function.

The FFmpeg and FFprobe host requirements are declared in
`environments/tools.toml` and checked with `evalkit env check tools`. The
frame policy remains in the user integration's own config.

## Common Failures

- A `file://` request fails while inline succeeds: the server cannot see the same
  path or its allowed-media root is too narrow.
- Video works but audio is ignored: confirm `prompt.audio` is on and the backend
  accepts `audio_url` for the selected model.
- Agent tool turns fail: preserve `tool_calls` on assistant messages and
  `tool_call_id` on tool responses.
- Scores differ after changing frame policy: treat it as a different evaluation
  condition and use a separate output directory.
