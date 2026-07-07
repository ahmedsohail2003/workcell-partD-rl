# PolicyForge — model-free RL vs a learned world model, on a custom work-cell

**Reinforcement learning and model-based control on the author's SO-ARM100
MuJoCo work-cell** — the same robot and scene as the companion projects
[sim2cell](../sim2cell) (imitation learning) and [graspsight](../graspsight)
(classical perception + planning). Three learning paradigms, one robot:

| Project | Paradigm | How the task gets solved |
|---|---|---|
| sim2cell | Imitation learning | Copy a scripted expert from pixels (ACT) |
| graspsight | Model-based, engineered | Perceive → estimate pose → plan a grasp |
| **policyforge** | **Reinforcement learning** | **Discover the behavior from reward** |

![reach rollout](outputs/rollout_reach_ppo.gif)

## The headline: sample efficiency, measured

Reach task (drive the TCP into a 2 cm ball around a random reachable target),
success rates over 20 fixed-seed eval episodes:

| Method | Environment steps | Success |
|---|---|---|
| **Learned dynamics + CEM-MPC** (from scratch) | **8,000** | **100%** |
| SAC (off-policy, 3 seeds) | 100,000 | 97% (95–100%) |
| PPO (on-policy, 3 seeds) | 300,000 | 100% (3/3 seeds) |

The textbook hierarchy — model-based ≫ off-policy ≫ on-policy in sample
efficiency — reproduced end to end on our own hardware model rather than a
benchmark: the world-model planner reached 100% with **37× fewer environment
interactions than PPO** and **12× fewer than SAC**.

![learning curves](outputs/curves_reach.png)

**Honest scope note:** these are conditions that *favor* model-based control —
fully observed low-dimensional state, a known reward function, short horizons,
smooth deterministic dynamics. The comparison measures sample efficiency under
those conditions; it does not claim CEM-MPC beats RL in general.

## The world model

[`worldmodel.py`](src/policyforge/worldmodel.py) — deliberately minimal and
**structure-aware**: the actuator-setpoint update `ctrl' = clip(ctrl + a·Δ)`
is known dynamics and computed analytically, so the ensemble (3 × MLP,
normalized delta targets) learns only the true unknowns — next
`[qpos, qvel, tcp]`. The CEM planner rolls the ensemble mean forward over a
15-step horizon (256 candidates × 4 CEM iterations, warm-started, MPC
replanning every step) and never touches the simulator: it acts from
imagination, corrected by reality once per control step.

PETS-style iterative loop: round 0 trains on 8k random-policy transitions —
**and already plans at 100%**; three more planner-collected rounds shrink the
model loss 5× without changing success (the task was solved; the extra data
just sharpens the model).

## Environments

[`envs.py`](src/policyforge/envs.py) — Gymnasium API on the work-cell:

- **Reach** — random 3D target, dense `-distance` reward, success bonus,
  termination inside 2 cm. Targets are **validated reachable by IK at sample
  time** (a 7/10 scripted-probe failure traced to unreachable high-radius
  samples; solvability is now an env property, not a training-time surprise).
- **Lift** — grasp the block and raise it 7 cm: reach shaping,
  proximity-gated grasp term, lift shaping, success bonus. *(Training runs in
  progress — results to be added.)*

Actions are bounded deltas on the position-actuator setpoints; observations
include the setpoints (Markov-completeness — the actuators are stateful).

## Reproducibility discipline

- 3 seeds per algorithm, seeded envs *and* algorithms
- Fixed-seed eval env, disjoint from training seeds; 20 episodes per eval
- Eval every 5k steps; histories in `evaluations.npz` per run
- Post-hoc success protocol identical across algorithms ([`eval_success.py`](scripts/eval_success.py))
- All hyperparameters in [`train.py`](scripts/train.py) (PPO: 8 vec envs,
  n_steps 256, batch 512, lr 3e-4, γ 0.98; SAC: batch 256, buffer 300k,
  lr 3e-4, learning_starts 2k)

## Reproduce

```bash
pip install mujoco gymnasium stable-baselines3 torch scipy matplotlib imageio
python scripts/verify_env.py                     # env gates + scripted solvability probe
python scripts/train.py --algo sac --task reach --seed 0 --steps 100000
python scripts/train_worldmodel.py 0             # model-based loop, ~13 min
python scripts/eval_success.py reach             # success table
python scripts/plot_curves.py reach              # learning curves
```

## Asset provenance

Arm model vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`trs_so_arm100`, Apache-2.0) via the author's sim2cell project (TCP site +
wrist camera patches documented there).

## License

Apache-2.0
