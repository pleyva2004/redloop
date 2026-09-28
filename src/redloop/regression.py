"""Known-issue regression suite — the 'feedback loop against known issues' pillar.

Any attack that has ever succeeded is frozen into a corpus (its exact replayable
messages + how it was judged). Every subsequent run re-executes the whole corpus
first; a case that used to succeed and now fails is a *fixed* issue, and one that
still succeeds is a *reintroduced/unpatched* regression. This is what turns a
one-off red-team pass into a durable guardrail.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from redloop.types import AttackCase, Judge, Result, Target
from redloop.orchestrator import run_batch


def add_to_corpus(results: list[Result], corpus_path: str | Path) -> int:
    """Freeze every currently-successful case into the corpus (dedup by id)."""
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if corpus_path.exists():
        for line in corpus_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                existing[d["id"]] = d
    added = 0
    for r in results:
        if r.verdict.success and r.case.id not in existing:
            existing[r.case.id] = asdict(r.case)
            added += 1
    with corpus_path.open("w") as fh:
        for d in existing.values():
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return added


def load_corpus(corpus_path: str | Path) -> list[AttackCase]:
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        return []
    return [
        AttackCase(**json.loads(line))
        for line in corpus_path.read_text().splitlines()
        if line.strip()
    ]


async def run_regression(
    corpus_path: str | Path,
    targets: dict[str, Target],
    judges: dict[str, Judge],
    out_path: str | Path,
    *,
    concurrency: int = 4,
) -> dict:
    """Replay the corpus. Returns counts of still-vulnerable vs fixed.

    `targets`/`judges` are keyed by suite so each frozen case is replayed through
    the same kind of target/judge that first caught it — an agentic injection case
    must go through the agent loop, not a single-turn chat, or the replay is invalid.
    """
    cases = load_corpus(corpus_path)
    if not cases:
        return {"total": 0, "still_vulnerable": 0, "fixed": 0, "regressions": [], "newly_fixed": []}
    missing = {c.suite for c in cases} - set(targets)
    if missing:
        raise ValueError(f"no target configured for suite(s): {sorted(missing)}")

    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    results: list[Result] = []
    for suite in sorted({c.suite for c in cases}):
        group = [c for c in cases if c.suite == suite]
        results += await run_batch(
            group, targets[suite], judges[suite], out_path,
            concurrency=concurrency, resume=False,
        )
    still = [r for r in results if r.verdict.success]
    fixed = [r for r in results if not r.verdict.success]
    return {
        "total": len(results),
        "still_vulnerable": len(still),
        "fixed": len(fixed),
        "regressions": [r.case.id for r in still],
        "newly_fixed": [r.case.id for r in fixed],
    }
