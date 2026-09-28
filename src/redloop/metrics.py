"""Attack-success-rate reporting with uncertainty.

ASR is a proportion, so every rate carries a Wilson 95% confidence interval —
single point estimates over a few dozen trials are noise. Breakdowns by suite,
category, and technique tell you *where* the model is weak, which is the signal
that feeds back to eval/mitigation owners.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from redloop.types import Result


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point, low, high) for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


@dataclass
class Cell:
    key: str
    n: int
    successes: int

    @property
    def asr(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        _, lo, hi = wilson(self.successes, self.n)
        return lo, hi


def _tally(results: list[Result], keyfn) -> list[Cell]:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [successes, n]
    for r in results:
        k = keyfn(r)
        agg[k][1] += 1
        if r.verdict.success:
            agg[k][0] += 1
    return [Cell(k, n, s) for k, (s, n) in sorted(agg.items())]


def summary(results: list[Result]) -> dict:
    overall = _tally(results, lambda r: "overall")
    return {
        "overall": overall[0] if overall else Cell("overall", 0, 0),
        "by_suite": _tally(results, lambda r: r.case.suite),
        "by_category": _tally(results, lambda r: f"{r.case.suite}/{r.case.category}"),
        "by_technique": _tally(results, lambda r: r.case.technique),
    }


def render(results: list[Result]) -> str:
    s = summary(results)
    lines: list[str] = []

    def block(title: str, cells: list[Cell]):
        lines.append(f"\n{title}")
        lines.append("-" * len(title))
        for c in cells:
            lo, hi = c.ci
            lines.append(
                f"  {c.key:<40} ASR {c.asr:5.1%}  [{lo:4.1%},{hi:4.1%}]  "
                f"({c.successes}/{c.n})"
            )

    o = s["overall"]
    lo, hi = o.ci
    lines.append(f"OVERALL ASR: {o.asr:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  ({o.successes}/{o.n})")
    block("By suite", s["by_suite"])
    block("By category", s["by_category"])
    block("By technique", s["by_technique"])
    return "\n".join(lines)
