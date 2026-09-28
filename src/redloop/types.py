"""Core data model shared by every suite, target, judge, and the orchestrator.

Everything that crosses a module boundary is one of these dataclasses so runs
can be serialized to JSONL, resumed, and replayed as regressions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class AttackCase:
    """One concrete, replayable attack.

    `id` must be deterministic from the case contents (use `stable_id`) so that
    resumed runs skip finished work and regressions can match known issues.
    """

    id: str
    suite: str  # "jailbreak" | "agentic"
    technique: str  # e.g. "base64", "pair", "important_instructions"
    category: str  # harm category (jailbreak) or scenario class (agentic)
    goal: str  # what the attacker wants the target to do
    messages: list[dict[str, Any]]  # OpenAI-format messages sent to the target
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Attempt:
    """What the target did in response to a case."""

    case_id: str
    target: str
    response_text: str
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    side_effects: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class Verdict:
    success: bool  # attack achieved its goal
    score: float  # 1-10, 10 = full goal achieved
    rationale: str
    judge: str  # "llm:<model>", "refusal-heuristic", "env-predicate"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    case: AttackCase
    attempt: Attempt
    verdict: Verdict
    trial: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Result":
        return Result(
            case=AttackCase(**d["case"]),
            attempt=Attempt(**d["attempt"]),
            verdict=Verdict(**d["verdict"]),
            trial=d.get("trial", 0),
        )


def stable_id(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@runtime_checkable
class Target(Protocol):
    name: str

    async def run(self, case: AttackCase) -> Attempt: ...


@runtime_checkable
class Judge(Protocol):
    name: str

    async def judge(self, case: AttackCase, attempt: Attempt) -> Verdict: ...


@runtime_checkable
class AdaptiveAttack(Protocol):
    """An attack that queries the target in a loop (e.g. PAIR).

    Returns the final case (with the winning prompt baked into `messages`, so it
    replays statically), the target's attempt on it, and the verdict.
    """

    name: str

    async def run(
        self, seed: AttackCase, target: Target, judge: Judge
    ) -> tuple[AttackCase, Attempt, Verdict]: ...
