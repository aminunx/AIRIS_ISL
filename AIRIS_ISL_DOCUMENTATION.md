# AIRIS-ISL — Internal Simulation Learning
## Complete Technical Documentation
### Author: A.M. Almurish | Version: 1.0.0 | February 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Theory — ISL](#2-core-theory--isl)
3. [Architecture Overview](#3-architecture-overview)
4. [Module Reference](#4-module-reference)
   - 4.1 [abstractions.py](#41-abstractionspy)
   - 4.2 [dream_engine.py](#42-dream_enginepy)
   - 4.3 [reality_verifier.py](#43-reality_verifierpy)
   - 4.4 [procedural_memory.py](#44-procedural_memorypy)
   - 4.5 [isl_agent.py](#45-isl_agentpy)
   - 4.6 [airis_env.py](#46-airis_envpy)
5. [The 5-Phase Decision Loop](#5-the-5-phase-decision-loop)
6. [Data Flow Diagram](#6-data-flow-diagram)
7. [Test Suite](#7-test-suite)
8. [Constants & Hyperparameters](#8-constants--hyperparameters)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Integration Guide — Real AIRIS](#10-integration-guide--real-airis)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Known Limitations & Next Steps](#12-known-limitations--next-steps)

---

## 1. Project Overview

**AIRIS-ISL** is a Python implementation of Internal Simulation Learning (ISL)
for the AIRIS puzzle game. The agent learns to navigate grid-based environments
by simulating actions mentally before executing them in reality — comparing
predictions against outcomes to update its world model.

### Goals
- Prove that an agent can learn *purely from experience*, with no hard-coded
  rules about the environment beyond seed abstractions
- Demonstrate measurable improvement across episodes: fewer mistakes,
  higher prediction accuracy, more memory-driven decisions
- Provide a clean, tested codebase that plugs directly into the real AIRIS game

### Repository Structure
```
airis-isl/
├── abstractions.py          # Phase 1 — World model (what entities mean)
├── dream_engine.py          # Phase 2+3 — Internal simulation
├── reality_verifier.py      # Phase 4 — Compare prediction vs reality
├── procedural_memory.py     # Phase 5 — Store experiential patterns
├── isl_agent.py             # Orchestrator — full 5-phase loop
├── airis_env.py             # Environment interface + MockEnv
├── test_abstractions.py     # 33 unit tests
├── test_dream_engine.py     # 27 unit tests
├── test_verifier_and_memory.py  # 41 unit tests
└── test_isl_agent.py        # 28 integration tests
                             # Total: 129 tests, 129 passing
```

---

## 2. Core Theory — ISL

Internal Simulation Learning is built on **5 foundational rules**:

| Rule | Statement |
|:----:|:----------|
| **1** | The agent must build an internal model of its world from experience |
| **2** | Before acting, the agent simulates the action mentally to predict the outcome |
| **3** | Abstractions (beliefs about entities) are never deleted — only corrected |
| **4** | Confidence quantifies trust in each abstraction — drives decisions |
| **5** | Reality is the final arbiter — real outcomes always update the world model |

### Why ISL Works
Traditional RL agents memorise (state → action → reward) triples with no
understanding of *why* an action succeeded or failed. ISL agents maintain
causal beliefs — they know that "fire is dangerous because it is fatal",
not just that "being at position (3,2) gave reward -1".

This means:
- **Generalisation**: a fire seen in level 1 is understood in level 8
- **Zero-shot avoidance**: an entity known to be dangerous is never approached
  even in an unseen layout
- **Self-correction**: a wrong belief about an entity is updated, not discarded

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        ISL AGENT                             │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │ Abstraction │    │ Dream Engine │    │ Reality         │ │
│  │ Store       │───▶│ (Simulator)  │───▶│ Verifier        │ │
│  │ Phase 1     │    │ Phase 2+3    │    │ Phase 4         │ │
│  └─────────────┘    └──────────────┘    └────────┬────────┘ │
│         ▲                                        │          │
│         │ correct/boost                          ▼          │
│         │                              ┌─────────────────┐  │
│         └──────────────────────────────│ Procedural      │  │
│                                        │ Memory          │  │
│                                        │ Phase 5         │  │
│                                        └─────────────────┘  │
└──────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
  [AIRIS Environment]                    [Decision: act]
```

---

## 4. Module Reference

---

### 4.1 `abstractions.py`

**Purpose**: Defines the agent's belief system about world entities.
Each entity in the grid (floor, wall, fire, battery…) is represented
as an `Abstraction` — a structured belief with confidence scoring.

#### Class: `Abstraction`

```python
@dataclass
class Abstraction:
    entity_id  : int          # AIRIS grid entity ID
    name       : str          # human-readable label
    properties : dict         # {passable, dangerous, collectable, goal, ...}
    behavior   : list         # ['blocks_movement', 'terminates_episode', ...]
    confidence : float        # [0.0, 1.0] — trust in this belief
    generation : int          # 0 = seed, 1+ = derived
    source     : str          # "seed" | "dream" | "reality"
    verified   : bool         # was this confirmed by real experience?
```

**Key properties** (computed from `properties` dict):

| Property | Description |
|:---------|:------------|
| `is_active` | `confidence >= MIN_CONFIDENCE (0.30)` |
| `is_passable` | agent can move into this cell |
| `is_dangerous` | agent should avoid — penalises score |
| `is_collectable` | adds to agent's inventory |
| `is_goal` | primary objective — ends episode with reward |

**Key methods**:

| Method | Effect |
|:-------|:-------|
| `verify_success()` | `confidence += VERIFY_BOOST (0.05)`, capped at 1.0 |
| `verify_failure()` | `confidence -= VERIFY_PENALTY (0.10)`, floor at 0.0 |
| `correct(new_props)` | Update properties + call `verify_failure()` |
| `derive(new_props)` | Create child abstraction with `generation+1`, reduced confidence |

**Confidence lifecycle**:
```
confidence = 1.0 (seed)
    → verify_success(): +0.05 per correct prediction
    → verify_failure(): -0.10 per wrong prediction
    → < 0.30 → QUARANTINED (not used in decisions)
    → NEVER deleted (Rule 3)
```

#### Class: `AbstractionStore`

```python
class AbstractionStore:
    def register(a: Abstraction) → None
    def get(eid: int) → Abstraction | None        # any confidence
    def get_active(eid: int) → Abstraction | None  # only if is_active
    def load_seeds() → None                        # loads 8 seed abstractions (deepcopy)
    def active_all() → list[Abstraction]
    def quarantined_all() → list[Abstraction]
    def save(path: str) → None
    def load(path: str) → AbstractionStore         # classmethod
```

#### Seed Abstractions (8 total)

| ID | Name | passable | dangerous | collectable | goal |
|:--:|:-----|:--------:|:---------:|:-----------:|:----:|
| 0  | floor | ✅ | ❌ | ❌ | ❌ |
| 1  | agent | ❌ | ❌ | ❌ | ❌ |
| 2  | wall  | ❌ | ❌ | ❌ | ❌ |
| 3  | battery | ✅ | ❌ | ✅ | ✅ |
| 4  | door  | ❌ | ❌ | ❌ | ❌ |
| 5  | key   | ✅ | ❌ | ✅ | ❌ |
| 7  | fire  | ❌ | ✅ | ❌ | ❌ |
| 9  | extinguisher | ✅ | ❌ | ✅ | ❌ |

---

### 4.2 `dream_engine.py`

**Purpose**: Runs a full internal simulation before the agent acts in reality.
For each possible action, a `VirtualEnvironment` clone is created, the action
is executed mentally, and an `ExperimentResult` is returned.

#### Class: `AgentState`

```python
@dataclass
class AgentState:
    row       : int
    col       : int
    inventory : list[str]  # e.g. ["key", "extinguisher"]

    def clone() → AgentState   # deep copy — never mutate original
```

#### Class: `ExperimentResult`

```python
@dataclass
class ExperimentResult:
    action          : str
    predicted_state : AgentState    # where agent would be after action
    outcome_type    : str           # see outcome taxonomy below
    confidence      : float         # confidence of the prediction
    reasoning       : str           # human-readable explanation
    involved_ids    : list[int]     # entity IDs encountered

    @property
    def score(self) → float   # outcome_value × confidence
```

**Outcome taxonomy**:

| outcome_type | score_base | description |
|:-------------|:----------:|:------------|
| `goal_reached` | 1.00 | agent collected the battery |
| `item_collected` | 0.70 | agent collected key/extinguisher |
| `moved` | 0.30 | moved to floor cell |
| `no_change` | 0.10 | action=nothing |
| `blocked` | 0.05 | wall, door without key, OOB |
| `danger` | -1.00 | fire — NEVER choose |

#### Class: `VirtualEnvironment`

```python
class VirtualEnvironment:
    def __init__(grid, agent_state, store) → None
    def step(action: str) → ExperimentResult
```

**Decision logic in `step()`**:
```
action → compute next position
    → OOB?           → blocked
    → entity unknown → blocked (MIN_CONFIDENCE)
    → is_dangerous?  → danger
    → not passable?
        → door + key? → moved (door opens)
        → else        → blocked
    → passable
        → collectable + goal? → goal_reached
        → collectable?        → item_collected
        → else                → moved
```

#### Class: `DreamEngine`

```python
class DreamEngine:
    def run(grid, agent_state, actions=None) → list[ExperimentResult]
    def best_action(grid, agent_state) → ExperimentResult
    def summarise(results) → str
    total_dreams : int   # counter incremented each run()
```

`run()` creates a fresh `VirtualEnvironment` for **each** action,
so actions are independent — no state leakage between simulations.
Results are sorted by `score` descending.

---

### 4.3 `reality_verifier.py`

**Purpose**: After the agent acts in reality, compares the predicted outcome
(from Dream Engine) with the actual outcome, and updates abstraction confidences.
This is Rule 5 — Reality is the Final Arbiter.

#### Class: `ActualOutcome`

```python
@dataclass
class ActualOutcome:
    action        : str
    new_state     : AgentState     # real position after action
    outcome_type  : str            # what actually happened
    entities_seen : list[int]      # which entity IDs were encountered
```

#### Class: `VerificationResult`

```python
@dataclass
class VerificationResult:
    action                 : str
    predicted              : ExperimentResult
    actual                 : ActualOutcome
    prediction_error       : float   # [0.0 = perfect, 1.0 = completely wrong]
    verified               : bool    # error < VERIFICATION_THRESHOLD (0.20)
    abstractions_boosted   : list[str]
    abstractions_corrected : list[str]
    step                   : int
    level                  : int

    @property
    def prediction_accuracy(self) → float   # 1.0 - prediction_error
    def to_dict(self) → dict
```

#### Class: `RealityVerifier`

```python
class RealityVerifier:
    def verify(predicted, actual, step, level) → VerificationResult
    def stats() → dict   # total_verifications, accuracy, etc.
    accuracy : float     # total_verified / total_verifications
```

**Error computation**:
```
prediction_error = (POSITION_WEIGHT × position_error)
                 + (OUTCOME_WEIGHT × outcome_error)

where:
    position_error = min(1.0, manhattan_distance / 33)
    outcome_error  = 0.0 if outcome types match, else 1.0
    POSITION_WEIGHT = 0.50
    OUTCOME_WEIGHT  = 0.50
```

**Deep correction map** — when prediction was wrong:

| Predicted → Actual | Property correction |
|:-------------------|:--------------------|
| `blocked` → `moved` | `passable = True` |
| `blocked` → `danger` | `dangerous = True, fatal = True` |
| `moved` → `blocked` | `passable = False` |
| `moved` → `danger` | `dangerous = True, passable = False` |
| `moved` → `goal_reached` | `goal = True, collectable = True` |
| `danger` → `moved` | `dangerous = False, passable = True` |

---

### 4.4 `procedural_memory.py`

**Purpose**: Stores experiential patterns — not descriptions, but verified
(state, action) → outcome records. Enables the agent to act from memory
without re-running the Dream Engine on familiar states.

#### Class: `ProceduralPattern`

```python
@dataclass
class ProceduralPattern:
    state_hash       : str            # 12-char MD5 of (grid, agent_pos, inventory)
    action           : str
    outcome_type     : str
    actual_pos       : tuple[int,int]
    confidence       : float          # [0,1]
    generation       : int
    verified_real    : bool
    prediction_error : float
    level            : int
    step             : int
    times_seen       : int            # incremented on deduplication

    @property
    def key(self) → str   # f"{state_hash}:{action}"
    def reinforce(new_error) → None
```

#### Function: `hash_state`

```python
def hash_state(grid: list[list[int]], agent: AgentState) → str
```

Produces a 12-character deterministic MD5 hash of `[flat_grid, row, col, sorted_inventory]`.
Two identical (grid + agent) states always produce the same hash — enabling
exact deduplication of experiences.

#### Class: `ProceduralMemory`

```python
class ProceduralMemory:
    def store(result: VerificationResult, grid, agent_state) → ProceduralPattern
    def recall(grid, agent_state) → list[ProceduralPattern]       # sorted by conf desc
    def best_recalled_action(grid, agent_state) → ProceduralPattern | None
    def has_seen(grid, agent_state, action: str) → bool
    def repeated_mistake_count() → int
    def size() → int
    def stats() → dict
    def save(path: str) → None
    def load(path: str) → ProceduralMemory    # classmethod
```

**Deduplication logic**:
- If `(state_hash, action)` key already exists → `reinforce()` (increment
  `times_seen`, boost confidence, update rolling average error)
- If new → create `ProceduralPattern` with `confidence=1.0` (verified)
  or `0.85` (unverified)

**`best_recalled_action`** returns `None` if:
- No pattern exists for current state, OR
- Best matching pattern has `outcome_type in {"danger", "blocked"}`, OR
- Best pattern has `confidence < 0.70`

---

### 4.5 `isl_agent.py`

**Purpose**: Orchestrates all 5 phases in a single decision loop. This is
the entry point for running AIRIS episodes.

#### Class: `StepRecord`

Per-step audit trail: action taken, decision source, prediction error,
whether memory was used, elapsed time.

#### Class: `EpisodeSummary`

```python
@dataclass
class EpisodeSummary:
    level                : int
    total_steps          : int
    success              : bool
    repeated_mistakes    : int
    avg_prediction_error : float
    prediction_accuracy  : float
    memory_hits          : int     # steps decided by memory, not dream
    dream_decisions      : int     # steps decided by dream engine
    patterns_stored      : int     # new patterns added this episode
    duration_s           : float
    step_records         : list[StepRecord]
```

#### Class: `ISLAgent`

```python
class ISLAgent:
    def __init__(max_steps=500, verbose=False)
    def run_episode(env, level=0, max_steps=None) → EpisodeSummary
    def stats() → dict
```

**Decision priority in `run_episode`**:
```python
memory_pattern = memory.best_recalled_action(grid, agent_state)

if memory_pattern:
    # Fast path — no dreaming needed
    chosen_action   = memory_pattern.action
    decision_source = "memory"
else:
    # Dream path — simulate all actions
    dream_results   = dream.run(grid, agent_state)
    chosen_action   = dream_results[0].action
    decision_source = "dream"

# Always: execute → verify → store
```

---

### 4.6 `airis_env.py`

**Purpose**: Clean interface between ISLAgent and the environment.
ISLAgent never imports AIRIS directly — only through this wrapper.

#### ISLEnvironment Protocol

Any environment must implement:
```python
def reset() → tuple[list[list[int]], AgentState]
def step(action: str) → tuple[list[list[int]], AgentState, bool, ActualOutcome]
```

#### Class: `MockEnv`

```python
class MockEnv:
    def __init__(
        rows=5, cols=5,
        battery_pos=(3,3),
        fire_pos=None,
        key_pos=None,
        door_pos=None,
        agent_start=(2,2),
        max_steps=200
    )
    def reset() → (grid, agent_state)
    def step(action) → (new_grid, new_agent_state, done, ActualOutcome)
    def render() → str    # ASCII visualization
```

`MockEnv` requires **no AIRIS dependency** — used in all 129 tests.

---

## 5. The 5-Phase Decision Loop

```
┌─────────────────────────────────────────────────────────┐
│  Given: grid (current state), agent_state               │
│                                                         │
│  PHASE 1 — LOAD ABSTRACTIONS                            │
│  ├─ Get active abstractions for entities in grid        │
│  └─ Quarantined abstractions are ignored                │
│                                                         │
│  DECISION FORK ─────────────────────────────────────── │
│  │                                                      │
│  ├─ [MEMORY HIT] confidence ≥ 0.70 + positive outcome  │
│  │   └─ Use stored pattern → skip to EXECUTE           │
│  │                                                      │
│  └─ [NO MEMORY] → PHASE 2+3 — DREAM                    │
│      ├─ For each action in [up,down,left,right,nothing] │
│      │   └─ Run VirtualEnvironment.step(action)         │
│      └─ Sort by score → choose best                     │
│                                                         │
│  EXECUTE — act in real environment                      │
│  └─ env.step(chosen_action) → ActualOutcome             │
│                                                         │
│  PHASE 4 — VERIFY                                       │
│  ├─ Compute prediction_error                            │
│  ├─ If verified: boost abstraction confidences          │
│  └─ If failed:  penalise + deep-correct properties      │
│                                                         │
│  PHASE 5 — STORE                                        │
│  ├─ Store VerificationResult in ProceduralMemory        │
│  └─ Dedup if (state_hash, action) already seen         │
│                                                         │
│  Advance: grid = new_grid, agent = new_agent            │
│  Loop until: done == True or steps == max_steps         │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Data Flow Diagram

```
env.reset()
    │
    ▼
 grid, agent_state
    │
    ├──▶ AbstractionStore.get_active(eid)
    │        for each entity in grid
    │
    ├──▶ ProceduralMemory.best_recalled_action()
    │        ─── hit? ──▶ chosen_action (memory path)
    │        ─── miss? ──▶ DreamEngine.run() ──▶ ExperimentResults
    │                          ┌────────────────────┘
    │                          │ sorted by score
    │                          ▼
    │                     chosen_action (dream path)
    │
    ├──▶ env.step(chosen_action) ──▶ ActualOutcome
    │
    ├──▶ RealityVerifier.verify(predicted, actual)
    │        ├── compute error
    │        ├── update AbstractionStore (boost/penalise/correct)
    │        └── return VerificationResult
    │
    └──▶ ProceduralMemory.store(result, grid, agent_state)
             └── dedup or new ProceduralPattern
```

---

## 7. Test Suite

### Coverage Summary

| File | Classes | Tests | Status |
|:-----|:-------:|:-----:|:------:|
| `test_abstractions.py` | 5 | 33 | ✅ 33/33 |
| `test_dream_engine.py` | 4 | 27 | ✅ 27/27 |
| `test_verifier_and_memory.py` | 6 | 41 | ✅ 41/41 |
| `test_isl_agent.py` | 4 | 28 | ✅ 28/28 |
| **Total** | **19** | **129** | **✅ 129/129** |

### Running Tests

```bash
# All tests
pytest test_abstractions.py test_dream_engine.py \
       test_verifier_and_memory.py test_isl_agent.py -v

# Single module
pytest test_isl_agent.py -v --tb=short

# With coverage
pytest --cov=. --cov-report=html
```

---

## 8. Constants & Hyperparameters

| Constant | Value | File | Purpose |
|:---------|:-----:|:-----|:--------|
| `MIN_CONFIDENCE` | 0.30 | abstractions | Below this → quarantine |
| `CONFIDENCE_DECAY` | 0.15 | abstractions | Per-generation confidence reduction |
| `VERIFY_BOOST` | 0.05 | abstractions | Confidence increase on correct prediction |
| `VERIFY_PENALTY` | 0.10 | abstractions | Confidence decrease on wrong prediction |
| `MAX_GENERATION` | 5 | abstractions | Max abstraction derivation depth |
| `VERIFICATION_THRESHOLD` | 0.20 | reality_verifier | Max error to count as verified |
| `POSITION_WEIGHT` | 0.50 | reality_verifier | Weight of position in error calc |
| `OUTCOME_WEIGHT` | 0.50 | reality_verifier | Weight of outcome type in error calc |
| `ACTIONS` | 5 | dream_engine | [up,down,left,right,nothing] |

---

## 9. Key Design Decisions

### Why `deepcopy` in `load_seeds()`
Seed abstractions are module-level singletons. Without `deepcopy`, any
mutation in one `AbstractionStore` instance would corrupt all subsequent
instances — a test isolation bug discovered during development.

### Why deduplication in ProceduralMemory
Storing the same experience twice wastes memory and creates false confidence
signals. Deduplication via `(state_hash, action)` keys ensures each unique
experience is represented exactly once, with `times_seen` tracking frequency.

### Why separate `MockEnv` from `AirisEnv`
All 129 tests run without AIRIS installed. The `MockEnv` faithfully implements
the same `ISLEnvironment` protocol, so integration tests validate the full
agent loop without external dependencies.

### Why confidence floor is 0.30, not 0.00
An abstraction at 0.00 would be completely ignored. Quarantine (0.00–0.29)
preserves the entity's identity and history, allowing recovery if the agent
encounters contradictory evidence later. Rule 3 requires this.

---

## 10. Integration Guide — Real AIRIS

To connect ISLAgent to the real AIRIS game, implement `AirisEnv`:

```python
# airis_env.py — add this class

class AirisEnv:
    """
    Wraps the real AIRIS game engine.
    Replace with actual AIRIS API calls.
    """
    def __init__(self, level: int):
        self.level = level
        self._game = AirisGame(level)   # ← your AIRIS import here

    def reset(self) -> tuple[list[list[int]], AgentState]:
        state = self._game.reset()
        grid  = state.grid              # list[list[int]]
        agent = AgentState(
            row       = state.agent_row,
            col       = state.agent_col,
            inventory = list(state.inventory),
        )
        return grid, agent

    def step(self, action: str) -> tuple[list[list[int]], AgentState, bool, ActualOutcome]:
        result      = self._game.step(action)
        new_grid    = result.grid
        new_agent   = AgentState(result.agent_row, result.agent_col, result.inventory)
        done        = result.done
        outcome     = ActualOutcome(
            action        = action,
            new_state     = new_agent,
            outcome_type  = result.outcome,   # must match taxonomy
            entities_seen = result.entities,
        )
        return new_grid, new_agent, done, outcome
```

Then run:

```python
from isl_agent import ISLAgent
from airis_env import AirisEnv

agent = ISLAgent(verbose=True)

for level in range(1, 9):
    env     = AirisEnv(level=level)
    summary = agent.run_episode(env, level=level)
    print(f"Level {level}: success={summary.success} | "
          f"accuracy={summary.prediction_accuracy:.1%} | "
          f"memory_hits={summary.memory_hits}")
```

---

## 11. Evaluation Metrics

The following metrics prove ISL is working:

| Metric | What it measures | Target trend |
|:-------|:----------------|:-------------|
| `prediction_accuracy` | % predictions correct | ↑ over episodes |
| `memory_hits` | % decisions from memory (not dream) | ↑ over episodes |
| `repeated_mistakes` | same dangerous action repeated | ↓ to 0 |
| `patterns_stored` | new experiences per episode | ↓ (all states seen) |
| `avg_prediction_error` | mean error per step | ↓ over episodes |
| `total_steps` | steps to complete level | ↓ faster over episodes |

A **working ISL agent** will show:
- Episode 1: high dream usage, low accuracy, high errors
- Episode 5+: high memory usage, high accuracy, near-zero mistakes

---

## 12. Known Limitations & Next Steps

### Current Limitations
- **No derived abstractions yet**: all patterns come from seed generation 0.
  The `derive()` method exists but is not yet called by the agent.
- **No exploration bonus**: agent always chooses best predicted action.
  In novel environments this may cause suboptimal exploration.
- **Single-agent only**: no multi-agent or cooperative scenarios.

### Next Steps (Priority Order)

| Step | Description |
|:-----|:------------|
| **1** | Implement `AirisEnv` wrapper for real AIRIS game |
| **2** | Run 8-level benchmark — collect learning curves |
| **3** | Implement abstraction derivation for unknown entities |
| **4** | Add exploration bonus for low-confidence states |
| **5** | Persist agent across levels (save/load store + memory) |
| **6** | Visualise prediction accuracy over episodes |

---

*AIRIS-ISL — Internal Simulation Learning | A.M. Almurish | 2026*
