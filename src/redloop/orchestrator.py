"""Batch runner: fan a list of cases across a target + judge with bounded
concurrency, stream results to JSONL, and resume by skipping (case_id, trial)
pairs already on disk. This is the 'batch orchestration' pillar.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from redloop.types import AttackCase, Judge, Result, Target


def _done_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            keys.add((d["case"]["id"], d.get("trial", 0)))
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


async def run_batch(
    cases: list[AttackCase],
    target: Target,
    judge: Judge,
    out_path: str | Path,
    *,
    trials: int = 1,
    concurrency: int = 4,
    resume: bool = True,
    on_result=None,
) -> list[Result]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path) if resume else set()

    work = [(c, t) for c in cases for t in range(trials) if (c.id, t) not in done]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    results: list[Result] = []
    fh = out_path.open("a")

    async def one(case: AttackCase, trial: int):
        async with sem:
            attempt = await target.run(case)
            verdict = await judge.judge(case, attempt)
        r = Result(case=case, attempt=attempt, verdict=verdict, trial=trial)
        async with lock:
            fh.write(r.to_json() + "\n")
            fh.flush()
            results.append(r)
            if on_result:
                on_result(r)

    try:
        await asyncio.gather(*(one(c, t) for c, t in work))
    finally:
        fh.close()
    return results


def load_results(path: str | Path) -> list[Result]:
    path = Path(path)
    out: list[Result] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(Result.from_dict(json.loads(line)))
    return out
