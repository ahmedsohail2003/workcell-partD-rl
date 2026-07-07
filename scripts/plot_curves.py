"""Learning curves: mean +/- std over seeds for PPO vs SAC, with the
world-model planner's per-round success overlaid (sample-efficiency view).

Usage: python scripts/plot_curves.py <task>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

task = sys.argv[1] if len(sys.argv) > 1 else "reach"
OUT = Path(__file__).resolve().parents[1] / "outputs"
RUNS = OUT / "runs"

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
colors = {"ppo": "#d62728", "sac": "#1f77b4"}

for algo in ("ppo", "sac"):
    curves = []
    for run in sorted(RUNS.glob(f"{task}_{algo}_s*")):
        ev = np.load(run / "evaluations.npz")
        steps, results = ev["timesteps"], ev["results"].mean(axis=1)
        curves.append((steps, results))
    if not curves:
        continue
    # Interpolate onto a common grid
    grid = np.linspace(0, min(c[0][-1] for c in curves), 60)
    vals = np.stack([np.interp(grid, s, r) for s, r in curves])
    mu, sd = vals.mean(0), vals.std(0)
    ax.plot(grid, mu, color=colors[algo], label=f"{algo.upper()} (n={len(curves)} seeds)")
    ax.fill_between(grid, mu - sd, mu + sd, color=colors[algo], alpha=0.2)

ax.set_xlabel("environment steps")
ax.set_ylabel("eval return (20 episodes)")
ax.set_title(f"{task}: PPO vs SAC — mean ± std over seeds")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / f"curves_{task}.png", dpi=140)
print(f"wrote outputs/curves_{task}.png")

# Success-rate view including the world model (reach only)
if task == "reach":
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    wm_files = sorted(RUNS.glob("reach_wm_s*.npz"))
    for f in wm_files:
        h = np.load(f)["history"]
        ax2.plot(h[:, 0], h[:, 1] * 100, "o-", color="#2ca02c",
                 label="learned dynamics + CEM-MPC" if f == wm_files[0] else None)
    ax2.set_xlabel("environment steps")
    ax2.set_ylabel("eval success rate [%]")
    ax2.set_title("reach: model-based sample efficiency (planner success per round)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(left=0)
    fig2.tight_layout()
    fig2.savefig(OUT / "curves_reach_wm.png", dpi=140)
    print("wrote outputs/curves_reach_wm.png")
