"""Console-script imports for source-only generators."""

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_console_script_loads_generator_from_working_directory(tmp_path):
    # A script outside cwd reproduces the installed console entry point's sys.path.
    script = tmp_path / "bin" / "evalkit"
    script.parent.mkdir()
    script.write_text("from evalkit.cli import main\nraise SystemExit(main())\n")
    work = tmp_path / "project"
    work.mkdir()
    (work / "local_agent.py").write_text(
        "from evalkit.types import GenerationResult\n"
        "async def generate(runtime, case, sampling):\n"
        "    return GenerationResult.completed('<answer>B</answer>')\n"
    )
    config = {
        "registry_path": str(REPO / "configs/registry.yaml"),
        "data_root": str(REPO / "data"),
        "out_dir": str(work / "results"),
        "model": {
            "generate_fn": "local_agent:generate",
            "endpoint": {"base_url": "http://127.0.0.1:9/v1", "model": "fixture"},
        },
        "eval_datasets": [{"name": "sample_mcq", "limit": 1}],
    }
    (work / "run.yaml").write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, str(script), "run", "--config", "run.yaml"],
        cwd=work,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    records = (work / "results/sample_mcq/predictions.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["response"] == "<answer>B</answer>"
