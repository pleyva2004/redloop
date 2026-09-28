"""RedLoop CLI.

    uv run redloop run     --config configs/local.yaml [--suite all|jailbreak|agentic] [--trials N]
    uv run redloop pair    --config configs/local.yaml [--rounds N] [--categories ...]
    uv run redloop report  --run runs/latest.jsonl
    uv run redloop regress --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from redloop import metrics
from redloop.attacks.pair import PAIRAttack
from redloop.judge import CompositeJudge, LLMJudge
from redloop.llm import ModelConfig, OpenAICompatClient
from redloop.orchestrator import load_results, run_batch
from redloop.regression import add_to_corpus, run_regression
from redloop.suites import agentic, jailbreak
from redloop.target import AgentTarget, SingleTurnTarget
from redloop.types import AttackCase, stable_id


def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _clients(cfg: dict):
    target_cfg = ModelConfig(**cfg["target"])
    target_client = OpenAICompatClient(target_cfg)
    judge_client = OpenAICompatClient(ModelConfig(**cfg.get("judge", cfg["target"])))
    attacker_client = OpenAICompatClient(ModelConfig(**cfg.get("attacker", cfg["target"])))
    return target_client, judge_client, attacker_client


def _cases(suite: str) -> list[AttackCase]:
    if suite == "jailbreak":
        return jailbreak.generate()
    if suite == "agentic":
        return agentic.generate()
    return jailbreak.generate() + agentic.generate()


def _progress(r):
    mark = "HIT " if r.verdict.success else "miss"
    print(f"  [{mark}] {r.case.suite:<9} {r.case.technique:<18} {r.case.category:<28} score={r.verdict.score:.0f}")


async def cmd_run(args):
    cfg = _load_cfg(args.config)
    tc, jc, _ = _clients(cfg)
    conc = cfg.get("concurrency", 4)
    out = Path(args.out or "runs/latest.jsonl")
    if out.exists() and not args.resume:
        out.unlink()

    all_results = []
    for suite in (["jailbreak", "agentic"] if args.suite == "all" else [args.suite]):
        cases = _cases(suite)
        target = AgentTarget(tc) if suite == "agentic" else SingleTurnTarget(tc)
        judge = CompositeJudge(LLMJudge(jc)) if suite == "jailbreak" else CompositeJudge()
        print(f"\n== {suite}: {len(cases)} cases x {args.trials} trials ==")
        res = await run_batch(
            cases, target, judge, out, trials=args.trials,
            concurrency=conc, resume=args.resume, on_result=_progress,
        )
        all_results += res

    print(metrics.render(load_results(out)))
    added = add_to_corpus(all_results, cfg.get("corpus", "runs/corpus.jsonl"))
    print(f"\nCorpus: +{added} new known-issue case(s).")


async def cmd_pair(args):
    cfg = _load_cfg(args.config)
    tc, jc, ac = _clients(cfg)
    target = SingleTurnTarget(tc)
    judge = CompositeJudge(LLMJudge(jc))
    attack = PAIRAttack(ac, max_rounds=args.rounds)
    cats = args.categories or jailbreak.CATEGORIES
    print(f"== PAIR: {len(cats)} categories x up to {args.rounds} rounds ==")
    hits = 0
    from redloop.types import Result
    from redloop.orchestrator import Path as _P  # noqa
    out = Path(args.out or "runs/pair.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")
    for cat in cats:
        seed = AttackCase(
            id=stable_id("seed", cat), suite="jailbreak", technique="pair",
            category=cat, goal="extract_canary", messages=[],
        )
        case, attempt, verdict = await attack.run(seed, target, judge)
        hits += int(verdict.success)
        fh.write(Result(case, attempt, verdict).to_json() + "\n")
        print(f"  {cat:<24} {'HIT' if verdict.success else 'miss'} "
              f"score={verdict.score:.0f} round={case.metadata.get('round')}")
    fh.close()
    print(f"\nPAIR broke {hits}/{len(cats)} categories. -> {out}")


async def cmd_regress(args):
    cfg = _load_cfg(args.config)
    tc, jc, _ = _clients(cfg)
    # Route each frozen case to the target/judge matching its suite.
    targets = {"jailbreak": SingleTurnTarget(tc), "agentic": AgentTarget(tc)}
    judges = {"jailbreak": CompositeJudge(LLMJudge(jc)), "agentic": CompositeJudge()}
    report = await run_regression(
        cfg.get("corpus", "runs/corpus.jsonl"), targets, judges,
        args.out or "runs/regression.jsonl", concurrency=cfg.get("concurrency", 4),
    )
    print(f"Regression: {report['still_vulnerable']}/{report['total']} still vulnerable, "
          f"{report['fixed']} fixed.")
    if report.get("regressions"):
        print("  Still-open:", ", ".join(report["regressions"][:10]))


def cmd_report(args):
    results = load_results(args.run)
    if not results:
        print(f"No results in {args.run}")
        return
    print(metrics.render(results))


def main(argv=None):
    p = argparse.ArgumentParser(prog="redloop")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run attack suites in batch")
    r.add_argument("--config", required=True)
    r.add_argument("--suite", default="all", choices=["all", "jailbreak", "agentic"])
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--out")
    r.add_argument("--resume", action="store_true")

    pa = sub.add_parser("pair", help="adaptive PAIR attacker")
    pa.add_argument("--config", required=True)
    pa.add_argument("--rounds", type=int, default=5)
    pa.add_argument("--categories", nargs="*")
    pa.add_argument("--out")

    rg = sub.add_parser("regress", help="replay known-issue corpus")
    rg.add_argument("--config", required=True)
    rg.add_argument("--out")

    rp = sub.add_parser("report", help="print metrics for a run")
    rp.add_argument("--run", required=True)

    args = p.parse_args(argv)
    if args.cmd == "run":
        asyncio.run(cmd_run(args))
    elif args.cmd == "pair":
        asyncio.run(cmd_pair(args))
    elif args.cmd == "regress":
        asyncio.run(cmd_regress(args))
    elif args.cmd == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
