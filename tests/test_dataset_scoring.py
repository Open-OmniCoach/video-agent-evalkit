"""Per-dataset scorer and aggregator overrides with no external calls.

Run:  python tests/test_dataset_scoring.py   (from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.bench import aggregate as agg
from evalkit.bench.scoring import DATASET_SCORERS, SCORERS, resolve_scorer
from evalkit.types import Prediction, ProblemType


def _diving48():
    scorer = DATASET_SCORERS["diving48_f16"]
    assert resolve_scorer(ProblemType.OPEN_ENDED, dataset="diving48_f16") is scorer

    gt = '{"takeoff":"Back","somersault":"15som","twist":"05Twis","flight":"FREE"}'
    sample = {"solution": gt, "problem_type": "open-ended"}

    def mk(raw):
        pred = scorer.extract(raw, sample)
        score = scorer.score(pred, sample)
        return pred, score

    pred_full, r_full = mk(gt)
    assert r_full == 1.0, r_full
    pred_partial, r_partial = mk('{"takeoff":"Back","somersault":"1som","twist":"05Twis","flight":"FREE"}')
    assert r_partial == 0.75, r_partial  # 3/4 phases

    records = [
        Prediction(uid="a", prediction=pred_full, score=r_full),
        Prediction(uid="b", prediction=pred_partial, score=r_partial),
    ]
    metrics = agg.aggregate(ProblemType.OPEN_ENDED, records, dataset="diving48_f16")
    assert metrics["protocol"] == "f16_phase_mean_accuracy", metrics
    assert metrics["exact_tuple_acc"] == 0.5, metrics  # 1 of 2 fully correct
    assert metrics["somersault_acc"] == 0.5, metrics  # only sample a matched
    assert metrics["takeoff_acc"] == 1.0, metrics
    assert abs(metrics["score"] - 0.875) < 1e-9, metrics  # (1.0 + 0.75)/2 phase-mean
    print("diving48_f16 OK:", {k: metrics[k] for k in ("score", "exact_tuple_acc", "somersault_acc")})


def _longvale():
    # Per-sample score is temporal IoU; aggregation reports mIoU and R1@IoU.
    records = [Prediction(uid=u, score=iou) for u, iou in [("x", 0.8), ("y", 0.4), ("z", 0.2)]]
    metrics = agg.aggregate(ProblemType.TEMPORAL_GROUNDING, records, dataset="longvale_grounding")
    assert metrics["protocol"] == "temporal_grounding_miou_r1", metrics
    assert abs(metrics["mIoU"] - (1.4 / 3)) < 1e-9, metrics
    assert abs(metrics["R1@0.3"] - 2 / 3) < 1e-9, metrics  # 0.8, 0.4 >= 0.3
    assert abs(metrics["R1@0.5"] - 1 / 3) < 1e-9, metrics  # only 0.8
    assert abs(metrics["R1@0.7"] - 1 / 3) < 1e-9, metrics
    print("longvale_grounding OK:", {k: round(metrics[k], 3) for k in ("mIoU", "R1@0.3", "R1@0.5")})


def _video_mme_v2():
    def group(group_id, group_type, structure, scores):
        return [
            Prediction(
                uid=f"{group_id}-{position}",
                score=score,
                facets={
                    "group_id": group_id,
                    "group_position": position,
                    "group_type": group_type,
                    "group_structure": structure,
                },
            )
            for position, score in enumerate(scores, 1)
        ]

    records = group("a", "relevance", "[1,2,3,4]", [1, 1, 1, 1])
    records += group("b", "logic", "[1,2,3,4]", [1, 1, 0, 1])
    records.append(
        Prediction(
            uid="partial-1",
            score=1,
            facets={"group_id": "partial", "group_position": 1, "group_type": "relevance"},
        )
    )
    metrics = agg.aggregate(ProblemType.MULTIPLE_CHOICE, records, dataset="video_mme_v2_full")
    assert metrics["protocol"] == "official_video_mme_v2_grouped_rating", metrics
    assert metrics["n_groups"] == 2 and metrics["n_incomplete_groups"] == 1, metrics
    assert metrics["total_percent"] == 62.5, metrics
    assert metrics["score"] == 0.625, metrics
    assert metrics["relevance_score_percent"] == 100.0, metrics
    assert metrics["logic_score_percent"] == 25.0, metrics
    print("video_mme_v2 OK:", {key: metrics[key] for key in ("score", "n_groups", "total_percent")})


def _default_fallback():
    # No dataset override -> generic mean score and ProblemType scorer.
    records = [Prediction(uid="1", score=1.0), Prediction(uid="2", score=0.0)]
    metrics = agg.aggregate(ProblemType.MULTIPLE_CHOICE, records, dataset="unknown_bench")
    assert metrics == {"score": 0.5, "n_scored": 2}, metrics
    print("default fallback OK:", metrics)


def _failed_rows_count_as_zero():
    failed = Prediction(uid="failed", status="failed", facets={"topic": "math"})
    records = [
        Prediction(uid="correct", score=1.0, facets={"topic": "math"}),
        failed,
    ]
    metrics = agg.aggregate(
        ProblemType.MULTIPLE_CHOICE,
        records,
        dataset="unknown_bench",
        report={"breakdowns": [{"name": "topic", "key": "facets.topic"}]},
    )
    assert metrics["score"] == 0.5, metrics
    assert metrics["n_scored"] == 2, metrics
    assert metrics["breakdowns"]["topic"]["math"] == {
        "score": 0.5,
        "n_scored": 2,
    }, metrics

    diving = agg.aggregate(
        ProblemType.OPEN_ENDED,
        [Prediction(uid="ok", prediction=_parse_diving_gt(), score=1.0), failed],
        dataset="diving48_f16",
    )
    assert diving["score"] == 0.5, diving
    assert diving["n_scored"] == 2, diving

    grounding = agg.aggregate(
        ProblemType.TEMPORAL_GROUNDING,
        [Prediction(uid="ok", score=0.8), failed],
        dataset="longvale_grounding",
    )
    assert grounding["mIoU"] == 0.4, grounding
    assert grounding["n_scored"] == 2, grounding


def _parse_diving_gt():
    scorer = DATASET_SCORERS["diving48_f16"]
    value = '{"takeoff":"Back","somersault":"15som","twist":"05Twis","flight":"FREE"}'
    return scorer.extract(value, {"solution": value})


def _report_breakdowns():
    records = [
        Prediction(uid="1", score=1.0, facets={"topic": "math", "difficulty": "easy"}),
        Prediction(uid="2", score=0.0, facets={"topic": "math", "difficulty": "hard"}),
        Prediction(uid="3", score=1.0, facets={"topic": "science", "difficulty": "easy"}),
    ]
    metrics = agg.aggregate(
        ProblemType.MULTIPLE_CHOICE,
        records,
        report={
            "breakdowns": [
                {"name": "topic", "key": "facets.topic"},
                {"name": "topic_by_difficulty", "keys": ["facets.topic", "facets.difficulty"]},
            ]
        },
    )
    assert abs(metrics["score"] - (2 / 3)) < 1e-9, metrics
    assert metrics["breakdowns"]["topic"]["math"] == {"score": 0.5, "n_scored": 2}, metrics
    assert metrics["breakdowns"]["topic"]["science"] == {"score": 1.0, "n_scored": 1}, metrics
    assert metrics["breakdowns"]["topic_by_difficulty"]["math | hard"] == {
        "score": 0.0,
        "n_scored": 1,
    }, metrics
    print("report breakdowns OK")


def _explicit_override_typos_fail():
    try:
        resolve_scorer(ProblemType.OPEN_ENDED, scorer_name="missing_scorer")
    except KeyError as exc:
        assert "missing_scorer" in str(exc), exc
    else:
        raise AssertionError("missing explicit scorer should fail")

    records = [Prediction(uid="1", score=1.0)]
    assert agg.aggregate(ProblemType.MULTIPLE_CHOICE, records, dataset="unknown_bench") == {
        "score": 1.0,
        "n_scored": 1,
    }
    try:
        agg.aggregate(
            ProblemType.MULTIPLE_CHOICE,
            records,
            aggregator_name="missing_aggregator",
        )
    except KeyError as exc:
        assert "missing_aggregator" in str(exc), exc
    else:
        raise AssertionError("missing explicit aggregator should fail")

    try:
        agg.aggregate(
            ProblemType.MULTIPLE_CHOICE,
            records,
            aggregator_fn=lambda _: [],
        )
    except TypeError as exc:
        assert "expected dict" in str(exc), exc
    else:
        raise AssertionError("custom aggregator must return a dict")

    print("explicit override typo failures OK")


def _grounding_structured_solution():
    temporal = SCORERS[ProblemType.TEMPORAL_GROUNDING]
    temporal_pred = temporal.extract(
        '<answer>{"time":[1,3]}</answer>',
        {"problem_type": "temporal grounding"},
    )
    for solution in ('{"time":[1,3]}', {"time": [1, 3]}, [1, 3], (1, 3)):
        score = temporal.score(
            temporal_pred,
            {"problem_type": "temporal grounding", "solution": solution},
        )
        assert score == 1.0, (solution, temporal_pred, score)

    for raw in (
        '{"time":[1,3]',
        '{ "time": [1, 3]  \n   ',
        '{"time":[1.000000000000000000,3.000000000000000000',
        "<answer>From <t>1</t>s to <t>3</t>s</answer>",
        "<answer>[00:01,00:03]</answer>",
        "<answer>From <t>00:01</t>s to <t>00:03</t>s</answer>",
        '<observation>{"time":[9,10]}</observation><answer>{"time":[1,3]',
        "<observation>from 9 to 10</observation><answer>from 1 to 3",
        "<think><answer>from 9 to 10</think><answer>from 1 to 3</answer>",
    ):
        pred = temporal.extract(raw, {"problem_type": "temporal grounding"})
        score = temporal.score(
            pred,
            {"problem_type": "temporal grounding", "solution": '{"time":[1,3]}'},
        )
        assert pred == {"time": [1.0, 3.0]} and score == 1.0, (raw, pred, score)

    incomplete = temporal.extract(
        '{"time":[1,',
        {"problem_type": "temporal grounding"},
    )
    assert incomplete is None, incomplete

    for raw in (
        "<observation>Checked [00:01,00:03].</observation><answer>The event does not occur.</answer>",
        '<observation>{"time":[1,3]}</observation><answer>The event does not occur.</answer>',
        '<think>{"time":[1,3]}</think><answer> </answer>',
    ):
        pred = temporal.extract(raw, {"problem_type": "temporal grounding"})
        assert pred is None, (raw, pred)

    spatial = SCORERS[ProblemType.SPATIAL_TEMPORAL_GROUNDING]
    spatial_raw = '<answer>{"time":[1,3],"boxes":{"1":[0,0,10,10]}}</answer>'
    spatial_pred = spatial.extract(
        spatial_raw,
        {"problem_type": "spatial-temporal grounding"},
    )
    for solution in (
        '{"time":[1,3],"boxes":{"1":[0,0,10,10]}}',
        {"time": [1, 3], "boxes": {"1": [0, 0, 10, 10]}},
    ):
        score = spatial.score(
            spatial_pred,
            {"problem_type": "spatial-temporal grounding", "solution": solution},
        )
        assert score == 1.0, (solution, spatial_pred, score)

    print("grounding structured solution OK")


def _number_extraction():
    numerical = SCORERS[ProblemType.NUMERICAL]
    sample = {"problem_type": "numerical", "solution": "<answer>7</answer>"}
    for raw in ("7", "<answer>7</answer>", "The answer is 7", "approximately 7", "Final: 7 days"):
        pred = numerical.extract(raw, sample)
        score = numerical.score(pred, sample)
        assert pred == 7.0 and score == 1.0, (raw, pred, score)

    comma_sample = {"problem_type": "numerical", "solution": "1234"}
    pred = numerical.extract("The count is 1,234.", comma_sample)
    assert pred == 1234.0 and numerical.score(pred, comma_sample) == 1.0, pred

    regression = SCORERS[ProblemType.REGRESSION]
    regression_sample = {"problem_type": "regression", "solution": "100"}
    pred = regression.extract("about 95", regression_sample)
    assert pred == 95.0 and regression.score(pred, regression_sample) == 1.0, pred

    print("number extraction OK")


def _yes_no_extraction():
    scorer = SCORERS[ProblemType.SSR]
    no_sample = {"problem_type": "SSR", "solution": "No"}

    for raw in ("No", "<answer>No</answer>", "answer: no", "The answer is no.", "yesterday no"):
        pred = scorer.extract(raw, no_sample)
        score = scorer.score(pred, no_sample)
        assert pred == "NO" and score == 1.0, (raw, pred, score)

    pred = scorer.extract("annoyed", no_sample)
    assert pred == "", pred

    crr = SCORERS[ProblemType.CRR]
    crr_sample = {"problem_type": "CRR", "ovo_type": 1}
    pred = crr.extract("true", crr_sample)
    assert pred == "YES" and crr.score(pred, crr_sample) == 1.0, pred

    print("yes/no extraction OK")


def main() -> int:
    _diving48()
    _longvale()
    _video_mme_v2()
    _default_fallback()
    _failed_rows_count_as_zero()
    _report_breakdowns()
    _explicit_override_typos_fail()
    _grounding_structured_solution()
    _number_extraction()
    _yes_no_extraction()
    print("dataset scoring OK")
    return 0


def test_script() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
