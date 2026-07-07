"""Render a trained policy solving the task.

Usage:
    python scripts/rollout_gif.py outputs/runs/reach_sac_s0/best_model.zip reach
    python scripts/rollout_gif.py wm reach          # CEM-MPC planner (retrains quickly? no - loads nothing, plans with a fresh model is meaningless; wm mode replays with a trained ensemble if saved)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio.v3 as iio
import numpy as np

from policyforge.envs import LiftEnv, ReachEnv

model_path = sys.argv[1]
task = sys.argv[2] if len(sys.argv) > 2 else "reach"
OUT = Path(__file__).resolve().parents[1] / "outputs"

env = {"reach": ReachEnv, "lift": LiftEnv}[task]()
algo = "sac" if "sac" in model_path else "ppo"
from stable_baselines3 import PPO, SAC

model = (SAC if algo == "sac" else PPO).load(model_path, device="cpu")

frames = []
n_show = 3
shown = 0
ep = 0
while shown < n_show and ep < 20:
    obs, _ = env.reset(seed=9000 + ep)
    ep += 1
    ep_frames = []
    done_ok = False
    for _ in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        ep_frames.append(env.render())
        if term or trunc:
            done_ok = bool(info["success"])   # Reach terminates on success; Lift holds to truncation
            break
    if done_ok:
        frames.extend(ep_frames)
        frames.extend([ep_frames[-1]] * 5)
        shown += 1

name = f"rollout_{task}_{algo}.gif"
iio.imwrite(OUT / name, np.stack(frames), duration=1 / 15, loop=0)
print(f"wrote outputs/{name} ({shown} successful episodes, {len(frames)} frames)")
