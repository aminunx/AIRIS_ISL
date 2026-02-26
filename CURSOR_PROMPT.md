# AIRIS-ISL Cursor Prompt
# ═══════════════════════════════════════════════════════════════
# PROJECT: AIRIS — Internal Simulation Learning (ISL) Agent
# MISSION: Build a real agent that proves ISL theory works in practice
# AUTHOR : A.M. Almurish | VERSION: 1.0.0
# ═══════════════════════════════════════════════════════════════

---

## CONTEXT — What This Project Is

You are working on **AIRIS-ISL**: an agent that learns to play the AIRIS
puzzle game using Internal Simulation Learning (ISL).

ISL is a theory that an agent can learn purely from experience by:
1. Maintaining a world model (Abstraction Store)
2. Simulating actions mentally before executing them (Dream Engine)
3. Comparing predictions to reality and updating beliefs (Reality Verifier)
4. Remembering patterns from experience (Procedural Memory)
5. Using memory to skip simulation on familiar states (ISL Agent)

**This is not a standard RL project.** The agent does not learn reward
functions. It learns causal beliefs about world entities and uses them
to make decisions. The goal is to demonstrate measurable improvement
across episodes: higher prediction accuracy, more memory-driven decisions,
fewer mistakes — proving the theory works empirically.

---

## CODEBASE — Files You Are Working With

All files are in the same directory. Every file is fully tested (129 tests passing).

```
abstractions.py          — Phase 1: Abstraction + AbstractionStore + 8 seed entities
dream_engine.py          — Phase 2+3: VirtualEnvironment + DreamEngine + ExperimentResult
reality_verifier.py      — Phase 4: RealityVerifier + ActualOutcome + VerificationResult
procedural_memory.py     — Phase 5: ProceduralMemory + ProceduralPattern + hash_state
isl_agent.py             — Orchestrator: ISLAgent + EpisodeSummary + StepRecord
airis_env.py             — Environment interface: MockEnv (no AIRIS dep) + AirisEnv stub
AIRIS_ISL_DOCUMENTATION.md — Full technical documentation (read this first)
test_abstractions.py      — 33 tests
test_dream_engine.py      — 27 tests
test_verifier_and_memory.py — 41 tests
test_isl_agent.py         — 28 integration tests
```

---

## RULES — Non-Negotiable Constraints

1. **NEVER delete abstractions** — only penalise confidence. Rule 3 of ISL.
2. **NEVER hard-code environment knowledge** — the agent learns from experience.
3. **Every new file must have tests** — minimum 10 tests per module.
4. **Run all 129 tests before and after every change** — zero regressions allowed.
5. **NEVER mutate the real grid** — only VirtualEnvironment clones.
6. **Always deepcopy seeds in load_seeds()** — singleton mutation corrupts tests.
7. **All confidence values must stay in [0.0, 1.0]** — enforced in __post_init__.
8. **The ISLEnvironment protocol is the ONLY interface to AIRIS** — no direct imports.

---

## IMMEDIATE TASK — Connect to Real AIRIS & Run Benchmark

### Step 1: Implement `AirisEnv` in `airis_env.py`

The `AirisEnv` class must implement the ISLEnvironment protocol:
```python
def reset() → tuple[list[list[int]], AgentState]
def step(action: str) → tuple[list[list[int]], AgentState, bool, ActualOutcome]
```

The grid uses these entity IDs (already defined in abstractions.py seeds):
- 0=floor, 1=agent, 2=wall, 3=battery, 4=door, 5=key, 7=fire, 9=extinguisher

The `outcome_type` in `ActualOutcome` MUST be one of:
`"moved" | "blocked" | "danger" | "goal_reached" | "item_collected" | "no_change"`

### Step 2: Run 8-Level Benchmark in `benchmark.py`

```python
# benchmark.py — create this file

from isl_agent import ISLAgent
from airis_env import AirisEnv   # or MockEnv for testing
import json

LEVELS    = range(1, 9)
EPISODES  = 5   # per level
MAX_STEPS = 500

agent   = ISLAgent(verbose=True)
results = []

for level in LEVELS:
    for episode in range(EPISODES):
        env     = AirisEnv(level=level)
        summary = agent.run_episode(env, level=level)
        results.append(summary.to_dict())
        print(f"L{level} E{episode+1}: "
              f"steps={summary.total_steps} | "
              f"accuracy={summary.prediction_accuracy:.1%} | "
              f"mem_hits={summary.memory_hits}")

# Save results
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Step 3: Generate Learning Curves in `plot_results.py`

Using `benchmark_results.json`, plot:
- X-axis: episode number (1 to 40 across all levels)
- Y-axis (left): `prediction_accuracy` per episode
- Y-axis (right): `memory_hits` per episode

**Expected result that proves ISL works:**
- prediction_accuracy starts ~0.50–0.70, reaches ~0.90+ by episode 20
- memory_hits starts at 0, grows to 50%+ by episode 10
- repeated_mistakes drops to 0 after episode 3

---

## WHAT COUNTS AS PROOF

The theory is proven if the benchmark shows ALL of the following:

| Metric | Episode 1 (expected) | Episode 20+ (target) |
|:-------|:--------------------:|:--------------------:|
| `prediction_accuracy` | < 0.70 | ≥ 0.90 |
| `memory_hits` | 0 | ≥ 30% of steps |
| `repeated_mistakes` | any | 0 |
| `avg_prediction_error` | > 0.30 | < 0.10 |
| `success` rate | any | 100% on seen levels |

If the agent fails to reach these targets, investigate:
1. Are abstraction confidences updating correctly? Check `RealityVerifier.stats()`
2. Is memory recall working? Check `ProceduralMemory.repeated_mistake_count()`
3. Is the Dream Engine scoring correctly? Print `dream.summarise(results)` per step

---

## CODE STYLE

- Python 3.10+ with type hints
- Dataclasses for all data containers
- No global state except `SEED_ABSTRACTIONS` (read-only after module load)
- All I/O (save/load) goes through explicit `path` parameters
- Test fixtures use `tmp_path` from pytest — never hardcoded paths
- Print statements only when `verbose=True` in ISLAgent
- Every class has `__repr__` returning meaningful info
- Module docstrings include: purpose, author, version

---

## DEBUGGING CHECKLIST

If tests fail after a change:

```bash
# 1. Run all tests — identify which module broke
pytest test_abstractions.py test_dream_engine.py \
       test_verifier_and_memory.py test_isl_agent.py -v --tb=short

# 2. If confidence tests fail — check VERIFY_BOOST/PENALTY constants
# 3. If isolation tests fail — check deepcopy in load_seeds()
# 4. If memory tests fail — check hash_state determinism
# 5. If integration tests fail — check ISLEnvironment protocol compliance
```

---

## FULL DOCUMENTATION

Read `AIRIS_ISL_DOCUMENTATION.md` before making any changes.
It contains: architecture diagrams, all class APIs, data flow,
decision logic, integration guide, and evaluation metrics.

---

## SUCCESS DEFINITION

The project is complete when:
1. `AirisEnv` connects to real AIRIS
2. `benchmark.py` runs 40 episodes across 8 levels without errors
3. `plot_results.py` generates a learning curve showing ISL improvement
4. The improvement curve matches the targets in "What Counts as Proof"
5. All 129 original tests still pass
6. A final report summarises: accuracy improvement, memory growth, mistake reduction

**This is not a demo. It is an empirical proof of a learning theory.**
Build it to be rigorous.
