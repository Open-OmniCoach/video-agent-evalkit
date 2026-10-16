# Subsets via annotation overlays

You often want to evaluate on a *subset* of a benchmark (e.g. a random 25% of
each of 5 benches) without touching the original data. evalkit supports this
with **zero code** using an **annotation overlay**: a lightweight `data_root`
override that changes only which annotation rows are loaded, keeping the
registered dataset id, registry entry, loader, scorer, and media layout
unchanged.

This is the recommended way to run subsets. Do not copy media into the subset.

## How evalkit resolves it

evalkit resolves a bench's paths like this:

- `registry_path` (run config) points at the **canonical** registry, e.g.
  `configs/registry.yaml`. It does not move.
- `data_root` (run config) is what a registry entry's **relative** `annotation`
  / `video_dir` / `image_dir` resolve against.

So an overlay = a directory that mirrors the registry's relative layout but
contains *filtered* annotation files (and a symlink back to the shared media).
Point `data_root` at it and the same benches load fewer rows. See
[`evalkit/bench/registry.py`](../evalkit/bench/registry.py) (`load_registry`,
`resolve_video_path`) and [data.md](data.md) for the canonical data-root
contract.

> Registry path convention: for overlays to mirror cleanly, register benches as
> `annotation: annotations/<name>.json` + `video_dir: benchmarks/<Name>`. (The
> shipped `configs/registry.yaml` uses a toy `examples/` layout for the demo
> bench.)

## Single-dataset overlay

```text
data/subsets/<dataset>__<selector>/
  annotations/<dataset>.json     # filtered/sampled rows, same schema + dataset id
  subset.json                    # provenance (evalkit ignores it; docs/contract only)
  benchmarks -> ../../benchmarks # symlink to shared media (do NOT copy)
```

Run config:

```yaml
registry_path: registry.yaml
data_root: ../data/subsets/tomato__random100-seed42
eval_datasets:
  - {name: tomato}
```

## Suite overlay (the "5 benches, 25% each" case)

Use a suite overlay when one run covers multiple benches. Each bench keeps its
**own** annotation file -- do **not** merge different datasets into one file.

```text
data/subsets/suite__my5__random25-seed42/
  annotations/
    bench1.json
    bench2.json
    bench3.json
    bench4.json
    bench5.json
  subset.json
  benchmarks -> ../../benchmarks
```

Run config:

```yaml
registry_path: registry.yaml
data_root: ../data/subsets/suite__my5__random25-seed42
out_dir: ../runs/my5_random25_seed42    # encode the selector so subsets don't clobber each other
eval_datasets:
  - {name: bench1}
  - {name: bench2}
  - {name: bench3}
  - {name: bench4}
  - {name: bench5}
```

Each bench is scored **separately** by its own scorer/aggregator, producing
`runs/.../bench1/summary.json`, etc., plus the top-level `summary.json`. There is
no cross-bench pooling (which would be semantically wrong across different
metrics like accuracy vs IoU).

## Constructing an overlay (data prep, run it yourself)

Sampling must be **deterministic** so the subset is reproducible, comparable
across models, and resume-safe. Materializing rows to a file already freezes the
subset; use a stable hash (order-independent) rather than `random.sample`:

```python
import hashlib, json, os
from pathlib import Path

def keep(uid: str, fraction: float, seed: int = 42) -> bool:
    h = hashlib.md5(f"{seed}:{uid}".encode()).hexdigest()
    return (int(h[:8], 16) % 10_000) < int(fraction * 10_000)

def make_overlay(overlay_dir, members, data_root, fraction=0.25, seed=42):
    overlay = Path(overlay_dir)
    (overlay / "annotations").mkdir(parents=True, exist_ok=True)
    # media symlink: no copy
    link = overlay / "benchmarks"
    if not link.exists():
        os.symlink(Path(data_root, "benchmarks").resolve(), link)
    components = []
    for name, ann_path in members.items():          # {bench: annotations/<bench>.json}
        rows = [json.loads(l) for l in Path(data_root, ann_path).read_text().splitlines() if l.strip()] \
               if ann_path.endswith(".jsonl") else json.loads(Path(data_root, ann_path).read_text())
        picked = [r for r in rows if keep(str(r.get("problem_id")), fraction, seed)]
        picked = picked or rows[:1]                  # ceil>=1 for tiny benches
        (overlay / "annotations" / f"{name}.json").write_text(json.dumps(picked, ensure_ascii=False))
        components.append({"dataset": name, "count": len(picked)})
    (overlay / "subset.json").write_text(json.dumps({
        "type": "suite_overlay", "selector": f"random{int(fraction*100)}-seed{seed}",
        "components": components,
        "selection": {"predicate": f"hash(problem_id) < {fraction}", "seed": seed},
    }, indent=2, ensure_ascii=False))
```

This is data preparation, not framework code -- evalkit only reads the resulting
annotation files. (A future `evalkit subset` helper could wrap this.)

## subset.json

Provenance / contract only; **evalkit does not read it**. Keep it small:

```json
{
  "type": "suite_overlay",
  "selector": "random25-seed42",
  "components": [{"dataset": "bench1", "count": 250}, {"dataset": "bench2", "count": 25}],
  "selection": {"predicate": "hash(problem_id) < 0.25", "seed": 42}
}
```

## Rules and edge cases

- **Determinism.** Sample by a seeded hash of `problem_id`; the materialized file
  then freezes the subset. Avoid unseeded randomness (breaks resume + cross-model
  comparability).
- **Per-selector out_dir.** Encode the selector in `out_dir` so different subsets
  (or seeds) do not share `predictions.jsonl` and cross-contaminate resume.
- **`data_root` is run-level.** All benches in a run share one `data_root`. To run
  some benches full and others subsetted in the same run, include the full ones in
  the overlay by symlinking their original annotation file (evalkit has no
  per-bench `data_root`).
- **Do not merge datasets into one file.** Keep one annotation file per bench (the
  dataset id ↔ loader/scorer contract depends on it). Merging into a single mixed
  benchmark is a different feature (a new dataset id + composite scoring) and is
  out of scope here.
- **Media is symlinked, never copied.** `video_dir` / `image_dir` resolve under
  `data_root`; the `benchmarks ->` symlink points back to the shared media tree.
- **Rounding / tiny benches.** Decide a rule (this doc uses ceil-to-≥1).
- **Selector names describe *selection*, not *evaluation*.** Put `fps` / sampling
  params / output dirs in the run config, not the subset id. Good:
  `suite__my5__random25-seed42`. Avoid: `suite__my5__fps1`.

## Boundary

An overlay **may**: filter/sample rows, preserve original fields + media paths,
add non-invasive provenance fields.

An overlay **must not**: modify `configs/registry.yaml`, copy media, encode
eval-only settings (fps, checkpoints, out dirs) in the subset id, or mix multiple
dataset schemas into one file. If media itself is clipped/transcoded/extracted,
make a separate data version instead.
