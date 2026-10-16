"""Data preparation tools.

Run:  python tests/test_data_tools.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evalkit.cli import main as evalkit_main
from evalkit.data_tools.links import ensure_overlay_benchmarks_symlink


def _write_dailyomni_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "Question": "Which event happened with the sound?",
                    "Choice": ["A. Door closes", "B. Music starts", "C. Dog barks"],
                    "Answer": "B",
                    "video_id": "vid001",
                    "Type": "AV Event Alignment",
                    "video_duration": "30s",
                }
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_lvomnibench_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "video_id": "video_222",
                    "video_link": "https://example.test/video",
                    "video_category": "Diy & Cooking",
                    "sub_category": "Cooking",
                    "duration": "26:22",
                    "question": "What is the relationship?",
                    "question_type": "Human-Centric Understanding",
                    "audio_type": "Speech",
                    "difficulty": "High",
                    "answer": "Mother",
                    "options": ["A. Wife", "B. Friend", "C. Mother", "D. Colleague"],
                    "correct_option": "C",
                    "question_id": "0000",
                }
            ],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_timelens_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "activitynet-timelens.json").write_text(
        json.dumps(
            {
                "v_keep": {
                    "duration": 12.5,
                    "spans": [[1, 3]],
                    "queries": ["A person opens a door."],
                },
                "v_missing": {
                    "duration": 7.0,
                    "spans": [[0, 2]],
                    "queries": ["This row has no local media."],
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (root / "charades-timelens.json").write_text("{}", encoding="utf-8")
    (root / "qvhighlights-timelens.json").write_text("{}", encoding="utf-8")
    media = root / "videos" / "activitynet" / "v_keep.mp4"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"")


def _write_worldsense_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "video_a": {
                    "video_id": "video_a",
                    "video_duration": "60s",
                    "duration": "<1min",
                    "domain": "Music",
                    "sub_category": "Covers",
                    "audio_class": ["Music"],
                    "video_caption": "caption",
                    "task0": {
                        "task_domain": "Understanding",
                        "task_type": "Spatial Relation",
                        "question": "Where is the object?",
                        "answer": "B",
                        "candidates": ["A. Left", "B. Right"],
                    },
                    "task1": {
                        "task_domain": "Recognition",
                        "task_type": "Object Counting",
                        "question": "How many objects?",
                        "answer": "A",
                        "candidates": ["A. One", "B. Two"],
                    },
                }
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_omnivideobench_source(path: Path) -> bool:
    try:
        import pandas as pd
    except ImportError:
        print("omnivideobench parquet fixture skipped: pandas unavailable")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "question": "Which sign is mentioned?",
                "question_type": "reference reasoning",
                "audio_type": "Speech",
                "reasoning_steps": [{"modality": "audio", "evidence": "spoken"}],
                "answer": "A sign",
                "options": ["A.A sign", "B.A song"],
                "correct_option": "A",
                "video": "videos/video_1.mp4",
                "video_type": "Vlog",
                "duration": "00:30",
            }
        ]
    )
    try:
        frame.to_parquet(path)
    except Exception as exc:  # noqa: BLE001 - optional parquet engine
        print(f"omnivideobench parquet fixture skipped: {type(exc).__name__}: {exc}")
        return False
    return True


def _test_dailyomni_convert_and_validate() -> None:
    with TemporaryDirectory() as tmp:
        data_root = Path(tmp) / "data"
        registry_path = Path(tmp) / "configs" / "registry.yaml"
        source = data_root / "benchmarks" / "Daily-Omni" / "qa.json"
        _write_dailyomni_source(source)

        rc = evalkit_main(
            [
                "data",
                "convert",
                "dailyomni",
                "--data-root",
                str(data_root),
                "--registry-path",
                str(registry_path),
                "--validate",
            ]
        )
        assert rc == 0, rc

        annotation = data_root / "annotations" / "dailyomni.json"
        rows = json.loads(annotation.read_text(encoding="utf-8"))
        assert rows[0]["problem_id"] == 1, rows
        assert rows[0]["solution"] == "<answer>B</answer>", rows
        assert rows[0]["path"] == "./vid001/vid001_video.mp4", rows
        assert rows[0]["data_source"] == "dailyomni", rows
        assert rows[0]["facets"] == {"task": "AV Event Alignment", "duration": "30s"}, rows

        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        assert registry["dailyomni"]["annotation"] == "annotations/dailyomni.json", registry
        assert registry["dailyomni"]["source_path"] == "benchmarks/Daily-Omni/qa.json", registry
        assert registry["dailyomni"]["video_dir"] == "benchmarks/Daily-Omni/Videos", registry
        assert registry["dailyomni"]["report"]["breakdowns"][0]["key"] == "facets.task", registry

        media = data_root / "benchmarks" / "Daily-Omni" / "Videos" / "vid001" / "vid001_video.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"")

        source.unlink()
        assert (
            evalkit_main(["data", "validate", "--data-root", str(data_root), "--registry-path", str(registry_path)])
            == 0
        )
        assert (
            evalkit_main(
                [
                    "data",
                    "validate",
                    "--data-root",
                    str(data_root),
                    "--registry-path",
                    str(registry_path),
                    "--check-source",
                ]
            )
            == 1
        )


def _test_omnivideobench_convert() -> None:
    with TemporaryDirectory() as tmp:
        data_root = Path(tmp) / "data"
        registry_path = Path(tmp) / "configs" / "registry.yaml"
        source = data_root / "benchmarks" / "OmniVideoBench" / "data.parquet"
        if not _write_omnivideobench_source(source):
            return

        rc = evalkit_main(
            [
                "data",
                "convert",
                "omnivideobench",
                "--data-root",
                str(data_root),
                "--registry-path",
                str(registry_path),
                "--validate",
            ]
        )
        assert rc == 0, rc

        rows = json.loads((data_root / "annotations" / "omnivideobench.json").read_text())
        assert rows[0]["path"] == "./videos/video_1.mp4", rows
        assert rows[0]["solution"] == "<answer>A</answer>", rows
        assert rows[0]["facets"]["task"] == "reference reasoning", rows
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        assert registry["omnivideobench"]["source_path"] == "benchmarks/OmniVideoBench/data.parquet"
        assert registry["omnivideobench"]["video_dir"] == "benchmarks/OmniVideoBench"


def _test_lvomnibench_convert() -> None:
    with TemporaryDirectory() as tmp:
        data_root = Path(tmp) / "data"
        registry_path = Path(tmp) / "configs" / "registry.yaml"
        source = data_root / "benchmarks" / "LVOmniBench" / "data.json"
        _write_lvomnibench_source(source)

        rc = evalkit_main(
            [
                "data",
                "convert",
                "lvomnibench",
                "--data-root",
                str(data_root),
                "--registry-path",
                str(registry_path),
                "--validate",
            ]
        )
        assert rc == 0, rc

        rows = json.loads((data_root / "annotations" / "lvomnibench.json").read_text())
        assert rows[0]["path"] == "./videos/video_222.mp4", rows
        assert rows[0]["solution"] == "<answer>C</answer>", rows
        assert rows[0]["facets"]["difficulty"] == "High", rows


def _test_timelens_convert_skips_missing_media() -> None:
    with TemporaryDirectory() as tmp:
        data_root = Path(tmp) / "data"
        registry_path = Path(tmp) / "configs" / "registry.yaml"
        source = data_root / "benchmarks" / "TimeLens-Bench"
        _write_timelens_source(source)

        rc = evalkit_main(
            [
                "data",
                "convert",
                "timelens_bench",
                "--data-root",
                str(data_root),
                "--registry-path",
                str(registry_path),
                "--validate",
            ]
        )
        assert rc == 0, rc

        rows = json.loads((data_root / "annotations" / "timelens_bench.json").read_text())
        assert len(rows) == 1, rows
        assert rows[0]["path"] == "./videos/activitynet/v_keep.mp4", rows
        assert rows[0]["problem_type"] == "temporal grounding", rows
        assert rows[0]["facets"] == {"source_dataset": "activitynet", "duration": 12.5}, rows


def _test_worldsense_convert() -> None:
    with TemporaryDirectory() as tmp:
        data_root = Path(tmp) / "data"
        registry_path = Path(tmp) / "configs" / "registry.yaml"
        source = data_root / "benchmarks" / "WorldSense" / "worldsense_qa.json"
        _write_worldsense_source(source)

        rc = evalkit_main(
            [
                "data",
                "convert",
                "worldsense",
                "--data-root",
                str(data_root),
                "--registry-path",
                str(registry_path),
                "--validate",
            ]
        )
        assert rc == 0, rc

        rows = json.loads((data_root / "annotations" / "worldsense.json").read_text())
        assert len(rows) == 2, rows
        assert rows[0]["path"] == "./video_a.mp4", rows
        assert rows[0]["solution"] == "<answer>B</answer>", rows
        assert rows[1]["task_id"] == "task1", rows
        assert rows[0]["facets"]["audio_class"] == ["Music"], rows


def _test_media_link_manifest() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / "data"
        common_root = root / "common"
        (common_root / "Daily-Omni").mkdir(parents=True)
        (common_root / "Video-MME" / "video").mkdir(parents=True)
        manifest = root / "media_links.yaml"
        manifest.write_text(
            "\n".join(
                [
                    f"common_root: {common_root}",
                    "links:",
                    "  benchmarks/Daily-Omni: Daily-Omni",
                    "  benchmarks/Video-MME/videos: Video-MME/video",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        rc = evalkit_main(["data", "link", "--data-root", str(data_root), "--manifest", str(manifest)])
        assert rc == 0, rc
        assert os.readlink(data_root / "benchmarks" / "Daily-Omni") == str(common_root / "Daily-Omni")
        assert os.readlink(data_root / "benchmarks" / "Video-MME" / "videos") == str(
            common_root / "Video-MME" / "video"
        )


def _test_overlay_benchmarks_symlink() -> None:
    with TemporaryDirectory() as tmp:
        overlay = Path(tmp) / "data" / "subsets" / "suite__demo__random25-seed42"
        ensure_overlay_benchmarks_symlink(overlay)
        assert os.readlink(overlay / "benchmarks") == "../../benchmarks"


def main() -> int:
    _test_dailyomni_convert_and_validate()
    _test_omnivideobench_convert()
    _test_lvomnibench_convert()
    _test_timelens_convert_skips_missing_media()
    _test_worldsense_convert()
    _test_media_link_manifest()
    _test_overlay_benchmarks_symlink()
    print("data tools OK")
    return 0


def test_script() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
