"""End-to-end evaluation runtime without an external model server.

Run: python tests/test_e2e_runtime.py
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evalkit.config import BenchCfg, load_run_config
from evalkit.generate import agent as agent_generate
from evalkit.runtime import driver
from evalkit.runtime.scoring import score as builtin_score
from evalkit.runtime.transport import EndpointPoolUnavailable
from evalkit.types import (
    ChatResult,
    EvalCase,
    GenerationResult,
    GenerationStatus,
    GenerationTiming,
    GenerationUsage,
    ProblemType,
    ScoreResult,
)

_GENERATE_CALLS: dict[str, int] = {}


async def flaky_generate(runtime, case, sampling):
    del runtime, sampling
    _GENERATE_CALLS[case.uid] = _GENERATE_CALLS.get(case.uid, 0) + 1
    if _GENERATE_CALLS[case.uid] == 1:
        return GenerationResult(status=GenerationStatus.FAILED, error="transient case failure")
    return GenerationResult.completed("<answer>B</answer>")


async def failing_score(runtime, case, generation):
    del runtime, case, generation
    raise RuntimeError("scoring unavailable")


async def custom_score(runtime, case, generation):
    del runtime, case
    return ScoreResult(prediction=generation.response, score=0.25)


def custom_aggregate(records):
    values = [float(record.score) for record in records]
    return {
        "score": sum(values) / len(values),
        "n_scored": len(values),
        "custom": True,
    }


async def unavailable_generate(runtime, case, sampling):
    del runtime, case, sampling
    raise EndpointPoolUnavailable("candidate pool is down")


async def partial_then_unavailable_generate(runtime, case, sampling):
    del runtime, sampling
    if case.uid == "mcq_2":
        await asyncio.sleep(0.02)
        raise EndpointPoolUnavailable("candidate pool went down")
    return GenerationResult.completed("<answer>B</answer>")


def execute_tool(tool_calls, case):
    del tool_calls, case
    return {"role": "tool", "tool_call_id": "call-1", "content": "7"}


def _last_text(messages) -> str:
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content
    return "".join(str(part.get("text", "")) for part in content or () if isinstance(part, dict))


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.released: list[str] = []
        self.active = 0
        self.max_active = 0
        self.agent_turns: dict[str, int] = {}
        self.__class__.instances.append(self)

    @classmethod
    def from_run_config(cls, config):
        del config
        return cls()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    async def chat(
        self,
        model_name,
        messages,
        *,
        sampling=None,
        request_body=None,
        route_key=None,
    ):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.005)
            text = _last_text(messages)
            self.calls.append(
                {
                    "model": model_name,
                    "messages": copy.deepcopy(messages),
                    "sampling": dict(sampling or {}),
                    "request_body": dict(request_body or {}),
                    "route_key": route_key,
                }
            )
            if model_name == "judge" or "You are grading" in text:
                answer = "1"
                message = {"role": "assistant", "content": answer}
            elif route_key == "agent-1":
                turn = self.agent_turns.get(str(route_key), 0)
                self.agent_turns[str(route_key)] = turn + 1
                if turn == 0:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    }
                    answer = ""
                else:
                    answer = "<answer>7</answer>"
                    message = {"role": "assistant", "content": answer}
            elif "2 + 2" in text:
                answer = "The answer is B"
                message = {"role": "assistant", "content": answer}
            elif "barks" in text:
                answer = "<answer>C</answer>"
                message = {"role": "assistant", "content": answer}
            elif "days" in text:
                answer = "Therefore, the final answer is: 7"
                message = {"role": "assistant", "content": answer}
            else:
                answer = ""
                message = {"role": "assistant", "content": answer}
            return ChatResult(
                message=message,
                text=answer,
                tool_calls=tuple(message.get("tool_calls") or ()),
                finish_reason="stop" if answer else "tool_calls",
                usage=GenerationUsage(prompt_tokens=10, completion_tokens=2),
                timing=GenerationTiming(latency_ms=5.0),
                endpoint="http://fake/v1/chat/completions",
            )
        finally:
            self.active -= 1

    def release_route_key(self, route_key):
        self.released.append(route_key)

    def metrics(self):
        return {"candidate": {"requests": len(self.calls)}}


def _run_config(out_dir: Path, bench: BenchCfg | None = None):
    config_path = out_dir.with_suffix(".yaml")
    config_path.write_text(
        json.dumps(
            {
                "registry_path": str(REPO / "configs/registry.yaml"),
                "data_root": str(REPO / "data"),
                "model": {"endpoint": {"base_url": "http://fake/v1", "model": "candidate"}},
                "defaults": {
                    "sampling": {"temperature": 0, "max_tokens": 512},
                    "scoring": {"type": "rule"},
                    "idempotent": True,
                },
                "retry": {"case_max_attempts": 2},
                "eval_datasets": [{"name": "sample_mcq"}],
            }
        )
    )
    config = load_run_config(config_path)
    return replace(
        config,
        out_dir=str(out_dir),
        max_active_cases=2,
        eval_datasets=(bench or config.eval_datasets[0],),
    )


def _test_accuracy_and_resume(root: Path) -> None:
    config = _run_config(root / "accuracy")
    result = driver.run_sync(config)
    summary = result["benchmarks"]["sample_mcq"]
    assert summary["n_total"] == 3, summary
    assert summary["n_failed"] == 0, summary
    assert summary["metrics"]["score"] == 1.0, summary
    assert summary["metrics"]["breakdowns"]["topic"]["math"]["score"] == 1.0
    runtime = FakeRuntime.instances[-1]
    assert len(runtime.calls) == 3, runtime.calls
    assert runtime.max_active <= 2, runtime.max_active
    assert set(runtime.released) == {"mcq_1", "mcq_2", "num_1"}, runtime.released

    rows = [
        json.loads(line) for line in (root / "accuracy" / "sample_mcq" / "predictions.jsonl").read_text().splitlines()
    ]
    assert {row["prediction"] for row in rows} == {"B", "C", 7.0}, rows
    assert all(row["generation"]["status"] == "completed" for row in rows), rows

    resumed = driver.run_sync(config)
    assert resumed["benchmarks"]["sample_mcq"]["metrics"]["score"] == 1.0
    assert FakeRuntime.instances[-1].calls == [], FakeRuntime.instances[-1].calls


def _test_retry_policy(root: Path) -> None:
    _GENERATE_CALLS.clear()
    retry_bench = BenchCfg(
        name="sample_mcq",
        generate_fn=f"{__name__}:flaky_generate",
        limit=1,
        idempotent=True,
    )
    result = driver.run_sync(_run_config(root / "retry", retry_bench))
    assert result["benchmarks"]["sample_mcq"]["metrics"]["score"] == 1.0, result
    assert _GENERATE_CALLS == {"mcq_1": 2}, _GENERATE_CALLS
    row = json.loads((root / "retry" / "sample_mcq" / "predictions.jsonl").read_text().strip())
    assert row["generation"]["attempts"] == 2, row
    assert row["generation"]["attempt_errors"] == ["transient case failure"], row

    _GENERATE_CALLS.clear()
    non_idempotent = replace(retry_bench, idempotent=False)
    result = driver.run_sync(_run_config(root / "no-retry", non_idempotent))
    assert result["benchmarks"]["sample_mcq"]["n_failed"] == 1, result
    assert result["benchmarks"]["sample_mcq"]["metrics"]["score"] == 0.0, result
    assert result["benchmarks"]["sample_mcq"]["metrics"]["n_scored"] == 1, result
    assert _GENERATE_CALLS == {"mcq_1": 1}, _GENERATE_CALLS


def _test_scoring_failure(root: Path) -> None:
    bench = BenchCfg(name="sample_mcq", score_fn=f"{__name__}:failing_score", limit=1)
    result = driver.run_sync(_run_config(root / "score-failure", bench))
    assert result["benchmarks"]["sample_mcq"]["n_failed"] == 1, result
    row = json.loads((root / "score-failure" / "sample_mcq" / "predictions.jsonl").read_text().strip())
    assert row["status"] == "failed", row
    assert row["response"] == "The answer is B", row
    assert row["generation"]["status"] == "completed", row
    assert row["score_details"] == {"failure_stage": "score"}, row


def _test_custom_scoring_and_aggregation(root: Path) -> None:
    bench = BenchCfg(
        name="sample_mcq",
        score_fn=f"{__name__}:custom_score",
        aggregator=f"{__name__}:custom_aggregate",
        limit=2,
    )
    result = driver.run_sync(_run_config(root / "custom-scoring", bench))
    metrics = result["benchmarks"]["sample_mcq"]["metrics"]
    assert metrics["score"] == 0.25, metrics
    assert metrics["n_scored"] == 2, metrics
    assert metrics["custom"] is True, metrics


def _test_endpoint_pool_failure_aborts_without_result_rows(root: Path) -> None:
    bench = BenchCfg(
        name="sample_mcq",
        generate_fn=f"{__name__}:unavailable_generate",
        limit=1,
    )
    out_dir = root / "pool-unavailable"
    try:
        driver.run_sync(_run_config(out_dir, bench))
    except EndpointPoolUnavailable:
        pass
    else:
        raise AssertionError("endpoint pool failure must abort the run")
    predictions = out_dir / "sample_mcq" / "predictions.jsonl"
    assert not predictions.exists() or not predictions.read_text().strip(), predictions


def _test_endpoint_pool_failure_flushes_completed_rows(root: Path) -> None:
    bench = BenchCfg(
        name="sample_mcq",
        generate_fn=f"{__name__}:partial_then_unavailable_generate",
        limit=2,
    )
    out_dir = root / "pool-unavailable-after-success"
    config = replace(_run_config(out_dir, bench), flush_every=20)
    try:
        driver.run_sync(config)
    except EndpointPoolUnavailable:
        pass
    else:
        raise AssertionError("endpoint pool failure must abort the run")
    predictions = out_dir / "sample_mcq" / "predictions.jsonl"
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert [row["uid"] for row in rows] == ["mcq_1"], rows


async def _agent_and_judge() -> None:
    runtime = FakeRuntime()
    case = EvalCase(
        uid="agent-1",
        dataset="agent-demo",
        index=0,
        messages=({"role": "user", "content": "Use a tool to find the number of days."},),
        target="<answer>7</answer>",
        problem_type=ProblemType.NUMERICAL,
        metadata={
            "budget": {"max_turns": 2},
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_executor": f"{__name__}:execute_tool",
        },
    )
    generated = await agent_generate.generate(runtime, case, {"temperature": 0})
    assert generated.status == GenerationStatus.COMPLETED, generated
    assert generated.response == "<answer>7</answer>", generated
    assert generated.usage.prompt_tokens == 20, generated.usage
    assert [event.kind for event in generated.trajectory].count("tool_call") == 1
    second_messages = runtime.calls[1]["messages"]
    assert second_messages[-1]["tool_call_id"] == "call-1", second_messages

    judged_case = replace(case, problem_type=ProblemType.OPEN_ENDED, target="seven")
    judged_generation = GenerationResult.completed("seven")
    judged = await builtin_score(
        runtime,
        judged_case,
        judged_generation,
        config={"type": "judge", "model": "judge"},
    )
    assert judged.score == 1.0 and judged.prediction == "seven", judged


def main() -> int:
    original_runtime = driver.EvalRuntime
    driver.EvalRuntime = FakeRuntime
    try:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _test_accuracy_and_resume(root)
            _test_retry_policy(root)
            _test_scoring_failure(root)
            _test_custom_scoring_and_aggregation(root)
            _test_endpoint_pool_failure_aborts_without_result_rows(root)
            _test_endpoint_pool_failure_flushes_completed_rows(root)
        asyncio.run(_agent_and_judge())
    finally:
        driver.EvalRuntime = original_runtime
    print("evaluation runtime OK: accuracy, resume, retry, failure stage, agent, judge")
    return 0


def test_script() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
