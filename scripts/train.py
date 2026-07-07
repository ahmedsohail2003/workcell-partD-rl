"""Train PPO or SAC on a work-cell task with a fixed eval protocol.

Reproducibility: explicit seed (env + algo), fixed-seed eval env, evals every
5k steps over 20 episodes, CSV history saved per run.

Usage:
    python scripts/train.py --algo ppo --task reach --seed 0 --steps 100000
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def make_env(task: str, seed: int, rank: int = 0):
    from policyforge.envs import LiftEnv, ReachEnv

    def _f():
        env = {"reach": ReachEnv, "lift": LiftEnv}[task]()
        env.reset(seed=seed + 1000 * rank)
        return env
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], required=True)
    ap.add_argument("--task", choices=["reach", "lift"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000)
    args = ap.parse_args()

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    run = REPO / "outputs" / "runs" / f"{args.task}_{args.algo}_s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)

    if args.algo == "ppo":
        venv = DummyVecEnv([make_env(args.task, args.seed, i) for i in range(8)])
        model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0, device="cpu",
                    n_steps=256, batch_size=512, learning_rate=3e-4, gamma=0.98)
    else:
        venv = DummyVecEnv([make_env(args.task, args.seed)])
        model = SAC("MlpPolicy", venv, seed=args.seed, verbose=0, device="cuda",
                    learning_rate=3e-4, batch_size=256, buffer_size=300_000,
                    learning_starts=2_000, gamma=0.98)

    eval_env = Monitor(make_env(args.task, args.seed + 7777)())
    cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=max(1, 5_000 // venv.num_envs),
                      best_model_save_path=str(run), log_path=str(run), verbose=0)

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    dt = time.perf_counter() - t0
    model.save(run / "final")

    ev = np.load(run / "evaluations.npz")
    mean_last = ev["results"][-1].mean()
    print(f"{args.task}/{args.algo}/seed{args.seed}: {args.steps} steps in {dt / 60:.1f} min, "
          f"final eval mean return {mean_last:.2f}")


if __name__ == "__main__":
    main()
