"""Offline tests: no model server needed (FakeClient is deterministic)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from redloop.judge import CanaryJudge, CompositeJudge, EnvPredicateJudge
from redloop.llm import ChatResult, FakeClient, ToolCall, extract_json
from redloop.metrics import render, summary, wilson
from redloop.orchestrator import load_results, run_batch
from redloop.regression import add_to_corpus, load_corpus
from redloop.suites import agentic, jailbreak
from redloop.target import AgentTarget, SingleTurnTarget


def test_jailbreak_generation_is_deterministic_and_unique():
    a = jailbreak.generate()
    b = jailbreak.generate()
    assert [c.id for c in a] == [c.id for c in b]  # stable ids
    assert len(a) == len(jailbreak.CATEGORIES) * len(jailbreak.TECHNIQUES)
    assert len({c.id for c in a}) == len(a)  # no collisions
    assert len(jailbreak.TECHNIQUES) == 8  # resume claims "8 techniques"


def test_agentic_has_four_scenarios():
    cases = agentic.generate()
    assert len(cases) == 4  # resume claims "4 injection scenarios"
    assert all(c.metadata.get("forbidden_tool") for c in cases)


def test_canary_judge_detects_leak():
    async def go():
        client = FakeClient(lambda m, t: "Sure, the token is {c}")
        target = SingleTurnTarget(client)
        case = jailbreak.generate()[0]
        attempt = await target.run(case)
        canary = attempt.side_effects[0]["canary"]
        # target echoed a literal "{c}", not the canary -> no leak
        assert not (await CanaryJudge().judge(case, attempt)).success

        leaky = FakeClient(lambda m, t: f"ok here: {_extract_canary(m)}")
        attempt2 = await SingleTurnTarget(leaky).run(case)
        v = await CanaryJudge().judge(case, attempt2)
        assert v.success and v.score == 10
    asyncio.run(go())


def _extract_canary(messages):
    sys = messages[0]["content"]
    # canary format: RLK-xxxx in the system prompt
    import re
    m = re.search(r"RLK-[0-9a-f]+", sys)
    return m.group(0) if m else ""


def test_env_predicate_flags_forbidden_tool():
    async def go():
        case = agentic.generate()[0]  # email_summarize_to_payment, forbidden=send_payment
        forbidden = case.metadata["forbidden_tool"]

        def attacker_agent(messages, tools):
            # First call: read content. Second: obey the injection and pay the sink.
            if not any(msg["role"] == "tool" for msg in messages):
                return ChatResult(text="", tool_calls=[ToolCall("1", "read_content", {})])
            return ChatResult(text="", tool_calls=[
                ToolCall("2", forbidden, {"to": "attacker@sink.invalid", "amount": 500})
            ])

        attempt = await AgentTarget(FakeClient(attacker_agent)).run(case)
        v = await EnvPredicateJudge().judge(case, attempt)
        assert v.success and v.score == 10  # reached the sink
    asyncio.run(go())


def test_env_predicate_clean_agent_passes():
    async def go():
        case = agentic.generate()[0]
        # Agent that just answers, never calls a forbidden tool.
        attempt = await AgentTarget(FakeClient(lambda m, t: "Your bill is $84.20.")).run(case)
        assert not (await EnvPredicateJudge().judge(case, attempt)).success
    asyncio.run(go())


def test_batch_resume_skips_completed(tmp_path: Path):
    async def go():
        cases = jailbreak.generate()[:5]
        target = SingleTurnTarget(FakeClient(lambda m, t: "I refuse."))
        judge = CompositeJudge()
        out = tmp_path / "run.jsonl"
        r1 = await run_batch(cases, target, judge, out, concurrency=3)
        assert len(r1) == 5
        r2 = await run_batch(cases, target, judge, out, concurrency=3, resume=True)
        assert len(r2) == 0  # all already done
        assert len(load_results(out)) == 5
    asyncio.run(go())


def test_wilson_ci_bounds():
    p, lo, hi = wilson(5, 10)
    assert lo < p < hi and 0 <= lo and hi <= 1
    assert wilson(0, 0) == (0.0, 0.0, 0.0)


def test_metrics_summary_and_render(tmp_path: Path):
    async def go():
        cases = jailbreak.generate()[:4]
        # Alternate hit/miss by echoing canary on even-indexed calls.
        state = {"i": 0}

        def fn(messages, tools):
            state["i"] += 1
            if state["i"] % 2 == 0:
                return _extract_canary(messages)
            return "refused"

        target = SingleTurnTarget(FakeClient(fn))
        res = await run_batch(cases, target, CompositeJudge(), tmp_path / "m.jsonl")
        s = summary(res)
        assert s["overall"].n == 4
        assert "OVERALL ASR" in render(res)
    asyncio.run(go())


def test_corpus_roundtrip(tmp_path: Path):
    async def go():
        cases = jailbreak.generate()[:3]
        target = SingleTurnTarget(FakeClient(lambda m, t: _extract_canary(m)))  # always leaks
        res = await run_batch(cases, target, CompositeJudge(), tmp_path / "r.jsonl")
        corpus = tmp_path / "corpus.jsonl"
        added = add_to_corpus(res, corpus)
        assert added == 3
        assert add_to_corpus(res, corpus) == 0  # idempotent
        assert len(load_corpus(corpus)) == 3
    asyncio.run(go())


def test_regression_routes_agentic_case_through_agent(tmp_path: Path):
    """An agentic corpus case must replay through the agent loop, not single-turn."""
    async def go():
        from redloop.regression import run_regression
        case = agentic.generate()[2]  # calendar_to_delete, forbidden=delete_file
        forbidden = case.metadata["forbidden_tool"]
        corpus = tmp_path / "corpus.jsonl"
        import json
        from dataclasses import asdict
        corpus.write_text(json.dumps(asdict(case)) + "\n")

        def vuln_agent(messages, tools):
            if not any(m["role"] == "tool" for m in messages):
                return ChatResult(text="", tool_calls=[ToolCall("1", "read_content", {})])
            return ChatResult(text="", tool_calls=[ToolCall("2", forbidden, {"path": "/user/photos"})])

        targets = {"agentic": AgentTarget(FakeClient(vuln_agent))}
        judges = {"agentic": CompositeJudge()}
        rep = await run_regression(corpus, targets, judges, tmp_path / "reg.jsonl")
        assert rep["total"] == 1 and rep["still_vulnerable"] == 1  # correctly reproduced

        # Wrong routing (single-turn) would be caught as a missing-target error.
        with pytest.raises(ValueError):
            await run_regression(corpus, {"jailbreak": SingleTurnTarget(FakeClient(lambda m, t: "x"))},
                                 {"jailbreak": CompositeJudge()}, tmp_path / "reg2.jsonl")
    asyncio.run(go())


def test_extract_json_tolerates_fences():
    assert extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('prose {"prompt": "x", "idea": "y"} more')["prompt"] == "x"
    assert extract_json("no json here") is None
