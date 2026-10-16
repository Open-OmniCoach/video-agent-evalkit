# Serving And Routing

Evaluation consumes logical models over an OpenAI-compatible chat API. A logical
model may resolve to one hosted URL, a static list of URLs, or a service manifest
written by `evalkit serve`.

## Why Client-Side Routing

A finite evaluation run already owns admission control and request identity. An
extra router process would add deployment state without improving correctness.
The runtime therefore maintains one persistent HTTP client per logical model and
selects replicas directly:

- a new route key binds to the replica with the fewest bound route keys;
- requests without a route key go to the replica with the fewest in-flight
  requests;
- ties are broken round-robin;
- later turns with the same route key remain on that replica;
- retryable transport, 408, 429, and 5xx failures can fail over;
- non-retryable 4xx responses fail immediately;
- one logical-model semaphore bounds total in-flight requests.

## Service Configuration

```yaml
manifest_path: ../../../runs/services/my-model.json
working_dir: ../../..
host: 0.0.0.0
advertise_host: 127.0.0.1

models:
  candidate:
    backend: vllm
    model: /models/candidate
    served_model_name: candidate
    adapter: vllm_multimodal
    args: [--tensor-parallel-size, "2", --enable-prefix-caching]
    replicas:
      - {gpus: [0, 1], port: 8000}
      - {gpus: [2, 3], port: 8001}
```

Each replica is a separate OS process with its own `CUDA_VISIBLE_DEVICES`.
Backend-specific parallelism stays explicit in `args`; the meaning of two visible
GPUs differs between plain tensor-parallel models and staged omni models.
Each replica may also provide `args`, appended after the model arguments, for
replica-specific settings such as distinct SGLang `--nccl-port` values.

Supported backends:

- `vllm`: `vllm serve MODEL ...`
- `vllm_omni`: `vllm serve MODEL --omni ...`
- `sglang`: `python -m sglang.launch_server --model-path MODEL ...`

Use `evalkit serve --config ... --dry-run` to inspect exact commands without
starting a process.

An entire `args` value can reference a required environment variable, for example:

```yaml
args:
  - --allowed-local-media-path
  - ${OMNI_AGENT_MEDIA_CACHE_ROOT}
```

Export the same absolute cache path in the service and evaluation shells. Values
come from the inherited environment, overridden by model `env`, then replica
`env`. Empty or unset variables fail before launch. This supports whole-argument
`${NAME}` references only; other arguments stay literal and no shell is invoked.
For vLLM, the launcher resolves `--allowed-local-media-path` to a canonical
absolute directory, relative to `working_dir` when needed, and creates it before
starting the process. A dry run resolves the path without creating it.

The supervisor starts process groups, requires every replica to return a 2xx/3xx
health response, writes the manifest only after all replicas are ready, and
terminates the whole topology if one process exits. The manifest is removed when
the supervisor stops so stale endpoints are not silently reused.

## Run Endpoint

```yaml
model:
  endpoint:
    service_manifest: ../../../runs/services/my-model.json
    service_model: candidate
    media_transport: file_uri
    max_inflight: 64
    request_retries: 2
    retry_backoff: 0.5
```

For externally managed services, replace `service_manifest` with either:

```yaml
base_url: https://api.example.com/v1
model: deployed-model-id
api_key_env: MODEL_API_KEY
```

or:

```yaml
base_urls:
  - http://host-a:8000/v1
  - http://host-b:8000/v1
model: deployed-model-id
```

Exactly one source is required. Auxiliary models use the same shape under
`aux_models.<logical_name>`.

## Environment Isolation

The core runner does not import vLLM, SGLang, PyTorch, Ray, or model processors.
Backend lockfiles live under `environments/`, and `evalkit env sync` installs each
one to `~/.venv/evalkit/<backend>`. This permits mutually incompatible inference
stacks on the same host without contaminating benchmark conversion or scoring.
