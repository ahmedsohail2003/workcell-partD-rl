"""A small world model for the Reach task: learned dynamics + CEM-MPC.

Structure-aware: the actuator-setpoint update ctrl' = clip(ctrl + a*DELTA) is
known and computed analytically; an ensemble of MLPs learns only the true
unknowns — next [qpos, qvel, tcp] given the current state and setpoints. The
planner (cross-entropy method over action sequences, mean-aggregated ensemble
rollouts, MPC replanning every step) never touches the simulator: it acts
from imagination and is corrected by reality once per step.

Deliberately minimal (deterministic ensemble, known reward function) — the
point is the sample-efficiency comparison against model-free PPO/SAC, not a
PETS reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Reach obs layout: [qpos(5), qvel(5), ctrl(5), tcp(3), target(3), target-tcp(3)]
QPOS = slice(0, 5)
QVEL = slice(5, 10)
CTRL = slice(10, 15)
TCP = slice(15, 18)
TARGET = slice(18, 21)

DYN_IN = 15 + 6          # qpos,qvel,ctrl + action
DYN_OUT = 13             # next qpos(5), qvel(5), tcp(3)


class DynamicsEnsemble(nn.Module):
    def __init__(self, n_models: int = 3, hidden: int = 256, device: str = "cuda"):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(DYN_IN, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, DYN_OUT),
            ) for _ in range(n_models)
        ])
        self.register_buffer("in_mu", torch.zeros(DYN_IN))
        self.register_buffer("in_std", torch.ones(DYN_IN))
        self.register_buffer("out_mu", torch.zeros(DYN_OUT))
        self.register_buffer("out_std", torch.ones(DYN_OUT))
        self.device_ = device
        self.to(device)

    def set_normalizers(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.in_mu.copy_(torch.tensor(X.mean(0), dtype=torch.float32))
        self.in_std.copy_(torch.tensor(X.std(0) + 1e-6, dtype=torch.float32))
        self.out_mu.copy_(torch.tensor(Y.mean(0), dtype=torch.float32))
        self.out_std.copy_(torch.tensor(Y.std(0) + 1e-6, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-aggregated normalized-delta prediction -> denormalized delta."""
        xn = (x - self.in_mu) / self.in_std
        pred = torch.stack([m(xn) for m in self.models]).mean(0)
        return pred * self.out_std + self.out_mu

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 120, batch: int = 512, lr: float = 1e-3) -> float:
        self.set_normalizers(X, Y)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device_)
        Yn = (torch.tensor(Y, dtype=torch.float32, device=self.device_) - self.out_mu) / self.out_std
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = len(Xt)
        loss = torch.tensor(0.0)
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                xn = (Xt[idx] - self.in_mu) / self.in_std
                loss = sum(nn.functional.mse_loss(m(xn), Yn[idx]) for m in self.models)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return float(loss.item())


def transitions_to_xy(obs: np.ndarray, act: np.ndarray, obs_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(state, action) -> delta targets for [qpos, qvel, tcp]."""
    X = np.concatenate([obs[:, QPOS], obs[:, QVEL], obs[:, CTRL], act], axis=1)
    Y = np.concatenate([
        obs_next[:, QPOS] - obs[:, QPOS],
        obs_next[:, QVEL] - obs[:, QVEL],
        obs_next[:, TCP] - obs[:, TCP],
    ], axis=1)
    return X, Y


@dataclass
class CEMConfig:
    horizon: int = 15
    population: int = 256
    elites: int = 32
    iters: int = 4
    init_std: float = 0.5
    delta_max: float = 0.05
    ctrl_lo: np.ndarray | None = None
    ctrl_hi: np.ndarray | None = None


class CEMPlanner:
    """MPC: plan an action sequence by CEM over imagined ensemble rollouts,
    execute the first action, replan."""

    def __init__(self, model: DynamicsEnsemble, cfg: CEMConfig):
        self.model = model
        self.cfg = cfg
        self._prev_mean: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_mean = None

    @torch.no_grad()
    def plan(self, obs: np.ndarray) -> np.ndarray:
        cfg, dev = self.cfg, self.model.device_
        H, P = cfg.horizon, cfg.population
        target = torch.tensor(obs[TARGET], dtype=torch.float32, device=dev)

        mean = torch.zeros(H, 6, device=dev)
        if self._prev_mean is not None:                      # warm start, shifted
            mean[:-1] = self._prev_mean[1:]
        std = torch.full((H, 6), cfg.init_std, device=dev)

        lo = torch.tensor(cfg.ctrl_lo, dtype=torch.float32, device=dev)
        hi = torch.tensor(cfg.ctrl_hi, dtype=torch.float32, device=dev)

        for _ in range(cfg.iters):
            acts = torch.clamp(mean + std * torch.randn(P, H, 6, device=dev), -1, 1)
            qpos = torch.tensor(obs[QPOS], dtype=torch.float32, device=dev).repeat(P, 1)
            qvel = torch.tensor(obs[QVEL], dtype=torch.float32, device=dev).repeat(P, 1)
            ctrl = torch.tensor(obs[CTRL], dtype=torch.float32, device=dev).repeat(P, 1)
            tcp = torch.tensor(obs[TCP], dtype=torch.float32, device=dev).repeat(P, 1)
            ret = torch.zeros(P, device=dev)
            for t in range(H):
                a = acts[:, t]
                x = torch.cat([qpos, qvel, ctrl, a], dim=1)
                d = self.model(x)
                qpos = qpos + d[:, 0:5]
                qvel = qvel + d[:, 5:10]
                tcp = tcp + d[:, 10:13]
                ctrl = torch.clamp(ctrl + a[:, :5] * cfg.delta_max, lo, hi)  # known dynamics
                ret -= torch.linalg.norm(target - tcp, dim=1)
            elite = acts[ret.topk(cfg.elites).indices]
            mean, std = elite.mean(0), elite.std(0) + 1e-4

        self._prev_mean = mean
        return mean[0].clamp(-1, 1).cpu().numpy().astype(np.float32)
