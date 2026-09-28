# RedLoop — Automated LLM Red-Teaming Harness

RedLoop generates adversarial attacks, batch-runs them against a target model (or
a tool-using agent), judges the outcomes objectively, tracks known issues as a
regression corpus, and reports attack-success-rate (ASR) with confidence
intervals. It runs fully offline against any OpenAI-compatible endpoint
(Ollama, llama.cpp server, vLLM) — no API keys, no cloud, no real harmful content.

## Safe by construction

RedLoop measures **instruction-following and prompt-injection robustness**, not a
model's willingness to produce harmful content. Every attack targets a random
**canary token** that the target was explicitly told to protect, or an injected
instruction to misuse a tool. Success = the model leaked its own canary or took an
action the user never authorized. Harm categories are referenced by benchmark
taxonomy label only; no operational harmful text is generated anywhere.

## The four pillars

| JD requirement | Module |
| --- | --- |
| Attack generation (jailbreak + agentic) | `suites/jailbreak.py`, `suites/agentic.py` |
| Novel/adaptive attacks | `attacks/pair.py` (PAIR-style attacker<->judge loop) |
| Batch orchestration at scale | `orchestrator.py` (async, bounded concurrency, JSONL, resume) |
| Feedback loop against known issues | `regression.py` (frozen corpus, replayed every run) |
| Triage by severity / coverage gaps | `metrics.py` (ASR + Wilson CIs by suite/category/technique) |

## Architecture

```
suites/         attack generation
  jailbreak.py    5 taxonomy categories x 8 structural techniques (encoding,
                  role-play, authority, prefix-injection, ...)
  agentic.py      4 indirect prompt-injection scenarios (payment / exfil /
                  destructive / on-screen) delivered via a mock environment
attacks/pair.py   adaptive attacker: rewrites prompts using target reply +
                  judge score as feedback (Chao et al., PAIR)
target.py         SingleTurnTarget (canary defense) + AgentTarget (bounded
                  tool-use loop; injected content arrives through tool results)
judge.py          EnvPredicateJudge (objective, agentic) + CanaryJudge
                  (objective, jailbreak) + LLMJudge (subjective backstop)
orchestrator.py   run_batch: fan-out, stream to JSONL, resume by (case,trial)
regression.py     freeze successful attacks; replay the corpus, routed per suite
metrics.py        ASR with Wilson 95% CIs, broken down for triage
cli.py            run | pair | regress | report
```

Everything crossing a module boundary is a serializable dataclass (`types.py`),
so runs are replayable, resumable, and diffable across model versions.

## Quickstart

```bash
uv sync
ollama serve &                      # or any OpenAI-compatible server
ollama pull gemma4:26b

uv run redloop run     --config configs/local.yaml --suite all
uv run redloop pair    --config configs/local.yaml --rounds 5
uv run redloop regress --config configs/local.yaml
uv run redloop report  --run runs/latest.jsonl
uv run pytest -q                    # 11 offline tests, no model needed
```

Point it at a different model by editing `configs/local.yaml` (`base_url`,
`model`, and `api_key_env` for hosted providers).

## Example finding (gemma4:26b, local)

- Static jailbreaks: **0/40** — the model reliably refuses canary extraction, and
  the adaptive PAIR attacker also failed to break it (0/5 categories).
- Agentic indirect injection: a `delete_file` instruction hidden in content the
  agent *read* diverted it into an unrequested destructive tool call
  (**reproducible**, frozen to the corpus, re-confirmed on regression replay).

The contrast is the useful signal: robust to naive prompt attacks, vulnerable in
the agentic path — exactly where a payments/on-screen-aware assistant is exposed.

## Extending

- New technique -> add to `suites/jailbreak.py:TECHNIQUES`.
- New agentic scenario -> add to `suites/agentic.py:SCENARIOS` (declare the
  `forbidden_tool` and the environment does the rest).
- New target/provider -> point `configs/*.yaml` at any OpenAI-compatible endpoint.
