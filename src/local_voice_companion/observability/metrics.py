"""Turn metrics aggregation.

One timeline per turn (core.events.TurnTimeline) is the raw record; this module
derives percentiles for the whole session so the UI can answer "how fast is it
really?" without inventing numbers.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..core.events import TurnTimeline


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for empty input."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


@dataclass
class TurnMetrics:
    total: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    samples: dict[str, list[float]] = field(default_factory=dict)

    def record(self, timeline: TurnTimeline, status: str = "completed") -> None:
        self.total += 1
        if status == "cancelled":
            self.cancelled += 1
        elif status == "failed":
            self.failed += 1
        else:
            self.completed += 1
        for name, value in timeline.metrics().items():
            self.samples.setdefault(name, []).append(float(value))

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total": self.total,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "failed": self.failed,
        }
        for name, values in self.samples.items():
            payload[name] = {
                "avg_ms": round(statistics.fmean(values), 1) if values else 0.0,
                "min_ms": round(min(values), 1) if values else 0.0,
                "max_ms": round(max(values), 1) if values else 0.0,
                "p50_ms": round(percentile(values, 50), 1),
                "p95_ms": round(percentile(values, 95), 1),
                "samples": len(values),
            }
        return payload

    def merge(self, other: "TurnMetrics") -> "TurnMetrics":
        merged = TurnMetrics(
            total=self.total + other.total,
            completed=self.completed + other.completed,
            cancelled=self.cancelled + other.cancelled,
            failed=self.failed + other.failed,
        )
        keys = set(self.samples) | set(other.samples)
        for key in keys:
            merged.samples[key] = list(self.samples.get(key, [])) + list(other.samples.get(key, []))
        return merged


class MetricsRegistry:
    """Bounded keep-last-N registry for recent turns."""

    def __init__(self, retain: int = 200) -> None:
        self.retain = max(1, retain)
        self.timelines: list[TurnTimeline] = []
        self.aggregate = TurnMetrics()

    def record(self, timeline: TurnTimeline, status: str = "completed") -> None:
        self.timelines.append(timeline)
        if len(self.timelines) > self.retain:
            del self.timelines[: len(self.timelines) - self.retain]
        self.aggregate.record(timeline, status)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.timelines[-limit:]]

    def summary(self) -> dict[str, Any]:
        return self.aggregate.summary()

    @staticmethod
    def from_timelines(timelines: Iterable[TurnTimeline]) -> dict[str, Any]:
        aggregate = TurnMetrics()
        for timeline in timelines:
            aggregate.record(timeline)
        return aggregate.summary()
