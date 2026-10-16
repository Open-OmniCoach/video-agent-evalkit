# Data Layout and Conversion

This document defines evalkit's data preparation contract. The evaluation
runtime must stay independent of upstream benchmark repositories and conversion
inputs: it reads only the registry, normalized annotations, and media paths.

## Canonical Layout

Use one `data_root` per prepared data tree:

```text
data/
  registry.yaml
  annotations/
    dailyomni.json
  benchmarks/
    Daily-Omni -> /shared/benchmarks/Daily-Omni
  subsets/
    suite__demo__random25-seed42/
```

All relative registry paths resolve against `data_root`.

```yaml
dailyomni:
  annotation: annotations/dailyomni.json
  video_dir: benchmarks/Daily-Omni/Videos
  status: ready
  source_format: json
  source_path: benchmarks/Daily-Omni/qa.json
  scoring_protocol: non_official_simple_mc_accuracy
  report:
    breakdowns:
      - {name: task, key: facets.task}
      - {name: duration, key: facets.duration}
```

Runtime fields:

| field | used by `evalkit run` | meaning |
|---|---:|---|
| `annotation` | yes | normalized `.json` / `.jsonl` samples |
| `video_dir` | yes | base directory for video sample `path` |
| `image_dir` | yes, for images | base directory for image sample `path` |
| `status` | yes | only `ready` entries load by default |
| `source_path` | no | optional raw converter input / provenance |
| `source_format` | no | optional converter/provenance note |
| `scoring_protocol` | no | human-readable contract note |
| `report` | yes | optional summary reporting config, e.g. `breakdowns` over annotation `facets` |

`source_path` is deliberately not a runtime dependency. If annotations and media
exist, evaluation should work even when raw converter inputs are absent. Use
`evalkit data validate --check-source` only when auditing a prepared data root
against its original conversion sources.

## Annotation Paths

Annotations are normalized evalkit rows. A video row typically looks like:

```json
{
  "problem_id": 1,
  "problem_type": "multiple choice",
  "problem": "Which event happened with the sound?",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "solution": "<answer>C</answer>",
  "data_type": "video",
  "path": "./abc123/abc123_video.mp4",
  "facets": {
    "task": "AV Event Alignment",
    "duration": "30s"
  }
}
```

For video samples, `path` is relative to the registry entry's `video_dir`. For
image samples, `path` is relative to `image_dir` when `image_dir` is configured.
The loader passes all annotation fields through `EvalCase.metadata`; scorers read
ground truth from `solution` or benchmark-specific metadata, not from
`source_path`.

`facets` is the normalized place for benchmark taxonomy used in reporting
breakdowns. Converters should map raw names such as `Type`, `category`,
`duration`, or `subject` into stable facet keys instead of making aggregators
understand every upstream schema.

## Media Symlinks

Prepared data roots should not copy benchmark media. Put shared media in one
common tree and expose the layout needed by each benchmark through symlinks.

Example manifest:

```yaml
common_root: ${EVALKIT_COMMON_BENCHMARKS}
links:
  benchmarks/Daily-Omni: Daily-Omni
  benchmarks/LVBench/videos: LVBench/video_chunks
  benchmarks/Video-MME/videos: Video-MME/video
```

Apply it with:

```bash
export EVALKIT_COMMON_BENCHMARKS=/mnt/path/to/shared/benchmarks
evalkit data link --data-root data --manifest configs/media_links.example.yaml
```

Rules:

- Link targets are resolved under `common_root` unless they are absolute paths.
- Declared targets must already exist; use a smaller manifest when only some
  benchmarks are available locally.
- Intermediate shim directories are allowed, e.g. `benchmarks/Video-MME/videos`
  can point at `common_root/Video-MME/video`.
- Existing non-symlink paths are rejected. Existing symlinks must already point
  at the expected target unless `--force` is passed.
- Environment variables in the manifest must resolve; unresolved `$VAR` paths
  are errors.

## Converters

Converters live under `evalkit/data_tools/converters/`, one file per benchmark.
They are data-preparation code, not runtime loaders. A converter should:

- read the benchmark's raw metadata from `source_path`;
- write normalized annotations under `annotations/<dataset>.json` or `.jsonl`;
- upsert a registry entry containing the runtime paths and optional provenance;
- validate normalized rows without importing runtime or upstream benchmark code.

Most converters should subclass `BaseConverter` from
`evalkit/data_tools/converters/base.py`. The base class handles source loading,
annotation writing, standard validation, structured registry entry generation,
and `facet_map` helpers. A benchmark converter typically only defines paths,
media directories, optional `report.breakdowns`, and `convert_row()`.

Daily-Omni is the first registered converter:

```bash
evalkit data convert dailyomni --data-root data --validate
```

By default it reads:

```text
data/benchmarks/Daily-Omni/qa.json
```

and writes:

```text
data/annotations/dailyomni.json
configs/registry.yaml
```

This assumes the Daily-Omni media tree is linked at
`data/benchmarks/Daily-Omni` and contains `qa.json` plus `Videos/`.
Its converter maps `Type` -> `facets.task` and `video_duration` ->
`facets.duration`, and its registry entry declares both as default breakdowns.

### Omni raw benchmark converters

Shared benchmark storage can be exposed without copying media:

```bash
export EVALKIT_COMMON_BENCHMARKS=/path/to/common_benchmarks
evalkit data link --data-root data --manifest configs/media_links.example.yaml
evalkit data convert omnivideobench omnivideo_test_full jointavbench_full \
  video_mme_full video_mme_v2_full futureomni_full \
  lvomnibench timelens_bench worldsense \
  --data-root data --validate
```

These converters read upstream/raw benchmark files directly:

| dataset | raw source | media base |
|---|---|---|
| `omnivideobench` | `benchmarks/OmniVideoBench/data.parquet` | `benchmarks/OmniVideoBench` |
| `omnivideo_test_full` | `benchmarks/OmniVideo-Test/test_505.jsonl` | `benchmarks/OmniVideo-Test` |
| `jointavbench_full` | `benchmarks/JointAVBench/jointavbench.json` | `benchmarks/JointAVBench` |
| `video_mme_full` | `benchmarks/Video-MME/videomme/test-00000-of-00001.parquet` | `benchmarks/Video-MME` |
| `video_mme_v2_full` | `benchmarks/Video-MME-v2/test.parquet` | `benchmarks/Video-MME-v2` |
| `futureomni_full` | `benchmarks/FutureOmni/futureomni_test.json` | `benchmarks/FutureOmni` |
| `lvomnibench` | `benchmarks/LVOmniBench/data.json` | `benchmarks/LVOmniBench` |
| `timelens_bench` | `benchmarks/TimeLens-Bench/*-timelens.json` | `benchmarks/TimeLens-Bench` |
| `worldsense` | `benchmarks/WorldSense/worldsense_qa.json` | `benchmarks/WorldSense` |

Expected row counts on the current shared data are:

```text
omnivideobench  1000 rows / 628 videos
omnivideo_test_full   505 rows / 264 source videos
jointavbench_full    2844 rows / 2844 clips
video_mme_full       2700 rows / 900 videos
video_mme_v2_full    3200 rows / 800 videos
futureomni_full      1034 rows / 1034 videos
lvomnibench     1014 rows / 275 videos
timelens_bench  9394 rows / 9394 queries
worldsense      3172 rows / 1662 videos
```

`timelens_bench` expands ActivityNet, Charades, and QVHighlights query/span
pairs. It skips raw rows whose local video file is absent; on the current
shared data this removes 10 ActivityNet queries from 3 missing videos.

`omnivideobench`, `video_mme_full`, and `video_mme_v2_full` read parquet, so
those converters require pandas with a parquet engine at data-preparation time.
Runtime evaluation does not import pandas.

The `_full` converters keep every question of the official test split; none
applies a duration filter. `jointavbench_full` applies the official seed-42
option shuffle to all 2,853 raw rows, then excludes the nine IDs the benchmark
marks as deprecated. `video_mme_v2_full` reports the official four-question
nonlinear group rating rather than per-question accuracy. Enable video audio
for `omnivideo_test_full`, `jointavbench_full`, and `futureomni_full`, and
disable it for the two Video-MME datasets.

## Validation

Validate the prepared runtime surface:

```bash
evalkit data validate --data-root data
```

This checks ready registry entries, annotation loading, `video_dir` / `image_dir`,
and a sample of media paths. It does not require `source_path`.

To audit conversion provenance too:

```bash
evalkit data validate --data-root data --check-source
```

To validate more media rows per dataset:

```bash
evalkit data validate --data-root data --media-sample 100
```

## Subsets

Subset data roots are annotation overlays: they mirror `annotations/` and link
`benchmarks -> ../../benchmarks`. They do not copy media and do not change the
canonical registry semantics. See [subsets.md](subsets.md).

## Boundary

The framework boundary is:

- `evalkit run` consumes normalized annotations and media paths only.
- Raw benchmark files are consumed only by `evalkit data convert`.
- `source_path` documents where annotations came from; it is not a label source.
- One benchmark's special conversion logic belongs in one converter module, not
  in the shared loader.
- A new benchmark should first try to fit the standard annotation schema before
  adding loader/scorer code.
