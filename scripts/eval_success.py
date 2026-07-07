"""Post-hoc success-rate evaluation of saved models: 20 fixed-seed episodes
per run, aggregated per algorithm. Produces the README results table.

Usage: python scripts/eval_success.py <task> [n_episodes]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from policyforge.envs import LiftEnv, ReachEnv

task = sys.argv[1] if len(sys.argv) > 1 else "reach"
n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
RUNS = Path(__file__).resolve().parents[1] / "outputs" / "runs"

from stable_baselines3 import PPO, SAC

env = {"reach": ReachEnv, "lift": LiftEnv}[task]()
summary: dict[str, list[float]] = {}
for run in sorted(RUNS.glob(f"{task}_*_s*")):
    if not (run / "best_model.zip").exists():
        continue
    algo = "sac" if "_sac_" in run.name else "ppo"
    model = (SAC if algo == "sac" else PPO).load(run / "best_model.zip", device="cpu")
    succ = 0
    for ep in range(n_eps):
        obs, _ = env.reset(seed=60_000 + ep)     # fixed eval seeds, disjoint from training
        for _ in range(env.max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term:
                succ += info["success"]
                break
            if trunc:
                break
    rate = succ / n_eps
    summary.setdefault(algo, []).append(rate)
    print(f"{run.name:<22} success {succ:>2}/{n_eps} ({rate * 100:3.0f}%)")

print()
for algo, rates in sorted(summary.items()):
    print(f"{task}/{algo.upper():<4} mean success over {len(rates)} seeds: "
          f"{np.mean(rates) * 100:.0f}% (min {np.min(rates) * 100:.0f}%, max {np.max(rates) * 100:.0f}%)")
