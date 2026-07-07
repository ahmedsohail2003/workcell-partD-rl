"""Model-based control on Reach: iterative data collection + CEM-MPC eval.

Round 0 collects random-policy transitions; each later round collects with the
current planner (plus exploration noise) and retrains — a minimal PETS-style
loop. Reports planner success rate per round against TOTAL env steps used,
the number that goes head-to-head with PPO/SAC sample efficiency.

Usage: python scripts/train_worldmodel.py [seed]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from policyforge.envs import ReachEnv
from policyforge.worldmodel import CEMConfig, CEMPlanner, DynamicsEnsemble, transitions_to_xy

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = Path(__file__).resolve().parents[1] / "outputs"
(OUT / "runs").mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed)
env = ReachEnv()
cfg = CEMConfig(ctrl_lo=env._ctrl_lo[:5], ctrl_hi=env._ctrl_hi[:5])
model = DynamicsEnsemble(device="cuda")
planner = CEMPlanner(model, cfg)

obs_buf, act_buf, next_buf = [], [], []
total_steps = 0
history = []


def collect(n_steps: int, policy: str, noise: float = 0.3) -> None:
    global total_steps
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    planner.reset()
    for _ in range(n_steps):
        if policy == "random":
            a = env.action_space.sample()
        else:
            a = planner.plan(obs)
            a = np.clip(a + rng.normal(0, noise, size=6).astype(np.float32), -1, 1)
        nobs, r, term, trunc, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        next_buf.append(nobs)
        total_steps += 1
        obs = nobs
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
            planner.reset()


def evaluate(n_eps: int = 20) -> tuple[float, float]:
    succ, dists = 0, []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=50_000 + ep)     # fixed eval seeds
        planner.reset()
        for _ in range(env.max_steps):
            obs, r, term, trunc, info = env.step(planner.plan(obs))
            if term:
                succ += 1
                break
            if trunc:
                break
        dists.append(info["distance"])
    return succ / n_eps, float(np.mean(dists))


t0 = time.perf_counter()
for rnd, (n_new, pol) in enumerate([(8_000, "random"), (3_000, "planner"), (3_000, "planner"), (2_000, "planner")]):
    collect(n_new, pol)
    X, Y = transitions_to_xy(np.array(obs_buf), np.array(act_buf), np.array(next_buf))
    loss = model.fit(X, Y)
    sr, md = evaluate()
    history.append((total_steps, sr, md))
    print(f"round {rnd}: env_steps={total_steps:>6}  fit_loss={loss:.4f}  "
          f"planner success={sr * 100:3.0f}%  mean final dist={md * 1000:.1f}mm", flush=True)

np.savez(OUT / "runs" / f"reach_wm_s{seed}.npz", history=np.array(history))
print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min; history saved")
