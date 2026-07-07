"""Verify the envs: SB3 API check, random rollouts, and a scripted solvability
probe for Reach (greedy IK-following controller must succeed — proves the task
is solvable and the success detector works before any RL runs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from stable_baselines3.common.env_checker import check_env

from policyforge.envs import LiftEnv, ReachEnv

for cls in (ReachEnv, LiftEnv):
    env = cls()
    check_env(env, warn=True)
    obs, _ = env.reset(seed=0)
    rews = []
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        rews.append(r)
        if term or trunc:
            env.reset()
    print(f"{cls.__name__:<9} obs={env.observation_space.shape}  act={env.action_space.shape}  "
          f"random-policy r/step: {np.mean(rews):+.3f} [{np.min(rews):+.3f}, {np.max(rews):+.3f}]")

# --- Reach solvability probe: greedy IK-following controller
from policyforge.ik import solve_ik  # noqa: E402

env = ReachEnv()
succ = 0
for ep in range(10):
    obs, _ = env.reset(seed=100 + ep)
    for _ in range(env.max_steps):
        q_t, _ = solve_ik(env.model, env.data, env.target)
        a5 = np.clip((q_t - env.data.ctrl[:5]) / env.DELTA_MAX, -1, 1)
        obs, r, term, trunc, info = env.step(np.concatenate([a5, [0.0]]).astype(np.float32))
        if term:
            succ += 1
            break
        if trunc:
            break
print(f"\nReach scripted probe: {succ}/10 (task solvable, success detector works)")
sys.exit(0 if succ >= 9 else 1)
