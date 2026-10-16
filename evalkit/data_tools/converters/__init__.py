"""Registered annotation converters."""

from __future__ import annotations

from . import (
    dailyomni,
    futureomni,
    jointavbench,
    lvomnibench,
    omnivideo_test,
    omnivideobench,
    timelens_bench,
    video_mme,
    video_mme_v2,
    worldsense,
)

CONVERTERS = {
    dailyomni.CONVERTER.name: dailyomni.CONVERTER,
    futureomni.CONVERTER.name: futureomni.CONVERTER,
    jointavbench.CONVERTER.name: jointavbench.CONVERTER,
    lvomnibench.CONVERTER.name: lvomnibench.CONVERTER,
    omnivideo_test.CONVERTER.name: omnivideo_test.CONVERTER,
    omnivideobench.CONVERTER.name: omnivideobench.CONVERTER,
    timelens_bench.CONVERTER.name: timelens_bench.CONVERTER,
    video_mme.CONVERTER.name: video_mme.CONVERTER,
    video_mme_v2.CONVERTER.name: video_mme_v2.CONVERTER,
    worldsense.CONVERTER.name: worldsense.CONVERTER,
}

__all__ = [
    "CONVERTERS",
    "dailyomni",
    "futureomni",
    "jointavbench",
    "lvomnibench",
    "omnivideo_test",
    "omnivideobench",
    "timelens_bench",
    "video_mme",
    "video_mme_v2",
    "worldsense",
]
