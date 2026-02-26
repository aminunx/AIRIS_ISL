# AIRIS_ISL Comprehensive Documentation

## 1. Purpose

`AIRIS_ISL` is an implementation of Internal Simulation Learning (ISL), where an agent learns by cycling through:

1. Wake perception (observe real state)
2. Dream simulation (simulate candidate actions internally)
3. Critic execution (pick best predicted action and execute in reality)
4. Verification (compare prediction vs actual outcome)
5. Selective integration (store validated patterns in procedural memory)

The environment is AIRIS (real game integration through `AirisEnv`), while the learning mind is implemented in the ISL modules.

## 2. Project Structure

- `abstractions.py`
  - Concept system (`Abstraction`, `AbstractionStore`)
  - Seed concepts (floor, wall, battery, fire, etc.)
  - Confidence updates and quarantine behavior

- `dream_engine.py`
  - Virtual one-step simulator (`VirtualEnvironment`)
  - Action simulation and scoring (`ExperimentResult`)
  - Goal-aware scoring with path distance

- `reality_verifier.py`
  - Prediction vs reality comparison
  - Error computation
  - Confidence boost/penalty and deep correction logic

- `procedural_memory.py`
  - State hashing and episodic procedural recall
  - Pattern storage and reuse

- `isl_agent.py`
  - Main orchestration of the 5 ISL phases
  - Decision loop, memory-first + dream fallback
  - Episode summary metrics

- `airis_env.py`
  - Adapter between ISL protocol and real AIRIS game engine

- `isl_learning_log.py`
  - Learning telemetry logger
  - Writes `isl_learning_log.json`
  - Writes discovery events to `isl_discoveries.json`

- `isl_visualizer.py`
  - Comparison runner (`ISL vs baseline`)
  - Produces aggregate metrics and result files

## 3. ISL Learning Cycle in This Codebase

### Phase 1: Wake (Perception)
- `ISLAgent.run_episode()` calls `env.reset()` and then receives current `(grid, agent_state)` every step.
- The agent loads active abstractions from visible entity IDs.

### Phase 2: Dream (Internal Simulation)
- `DreamEngine.run(grid, agent_state)` simulates all actions in a `VirtualEnvironment` copy.
- Fearless dreaming behavior:
  - `no_change` is strongly penalized.
  - Only truly dangerous outcomes are strongly negative.
- Goal direction:
  - Goal cell selected in virtual state.
  - BFS distance map is used to compute path-aware movement bonus.

### Phase 3: Critic (Action Selection)
- Dream results are score-sorted.
- Agent picks the top action (with safety guard against ending on `nothing` when alternatives exist).
- Procedural memory can short-circuit dreaming when reliable patterns exist.

### Phase 4: Verification (Reality Check)
- Real action is executed in `AirisEnv.step(action)`.
- `RealityVerifier.verify(predicted, actual)` computes `prediction_error` and updates abstractions:
  - Correct prediction: boost confidence
  - Incorrect prediction: penalize and correct relevant properties

### Phase 5: Selective Integration (Consolidation)
- `ProceduralMemory.store(...)` stores verified experience patterns.
- Repeated useful patterns become preferred for fast recall.

## 4. Dream Engine Path Awareness

`VirtualEnvironment` uses BFS-aware distance, not naive straight-line Manhattan only.
This prevents fake progress through walls and favors moves that reduce *real navigable* distance to the objective.

Key mechanisms:
- Goal discovery in virtual map
- BFS distance table from goal
- Per-action distance bonus added to score
- `no_change` strongly penalized to avoid passive loops

## 5. Learning Telemetry

`LearningTracker` (`isl_learning_log.py`) records per episode:
- `episode`, `level`, `steps`, `won`
- `avg_prediction_error`
- `avg_confidence`
- `danger_encountered`, `danger_avoided`
- `dream_count`, `memory_hit_count`
- `abstractions_updated`, `patterns_stored`

Files written:
- `isl_learning_log.json`: full episode stream
- `isl_discoveries.json`: discovery events (pattern creation / abstraction updates)

## 6. How to Run

### Install / environment
Use the Python environment configured for this project and ensure AIRIS dependencies are available.

### Run core tests
```bash
pytest test_abstractions.py test_dream_engine.py test_verifier_and_memory.py test_isl_agent.py -v
```

Expected baseline during current verified state:
- `109 passed`

### Run ISL vs baseline comparison
```bash
python isl_visualizer.py compare
```

### Show comparison summary
```bash
python isl_visualizer.py results
```

## 7. Reproducible Evaluation Workflow

1. Reset telemetry files (optional for clean run):
```bash
python -c "open('isl_learning_log.json','w',encoding='utf-8').write('[]'); open('isl_discoveries.json','w',encoding='utf-8').write('[]')"
```

2. Run compare:
```bash
python isl_visualizer.py compare
```

3. Quick learning sanity check:
```bash
python -c "import json; d=json.load(open('isl_learning_log.json')); e=[x['avg_prediction_error'] for x in d]; print('episodes',len(d)); print('first5',sum(e[:5])/5,'last5',sum(e[-5:])/5)"
```

4. Full proof-style report is generated in:
- `step4_proof.txt` (if proof script has been run)

## 8. Current Verified Outputs (Latest Local Run)

From latest recorded run artifacts:
- Tests: `109 passed`
- ISL vs baseline:
  - ISL win rate: `30/40 (75%)`
  - Baseline win rate: `10/40 (25%)`
- Prediction error trend (tracked metric): decreasing from early episodes to late episodes
- Discovery events: present (`isl_discoveries.json` non-empty)

These outcomes demonstrate measurable learning signals in the current implementation.

## 9. Important Constraints

- Do not break protocol contracts between modules.
- Keep ISL phases intact (wake -> dream -> critic -> verification -> integration).
- Always rerun full test suite after changes.
- Persist evaluation outputs for auditability.

## 10. Troubleshooting

### Agent loops or stalls
- Check `dream_engine.py` scoring weights and distance bonus behavior.
- Confirm `no_change` remains penalized.
- Inspect procedural memory dominance and recall thresholds.

### High prediction accuracy but low wins
- Validate goal detection and `goal_reached` outcome handling in `airis_env.py`.
- Inspect whether action selection is overusing repetitive safe actions.

### Inconsistent learning logs
- Ensure telemetry files are reset before benchmark runs when comparing first-vs-last episode trends.

## 11. Versioning and Audit Trail

Suggested commit discipline:
- Keep learning logic and telemetry changes in separate commits when possible.
- Save benchmark outputs (`step*.txt`, `isl_learning_log.json`) for reproducibility.
- Use detailed commit messages that include:
  - what changed
  - why it changed
  - how it was validated

---

Maintained in repository root as a single source of truth for architecture, operation, and evaluation.
