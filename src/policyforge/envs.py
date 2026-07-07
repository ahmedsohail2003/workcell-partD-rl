"""Gymnasium environments on the SO-ARM100 work-cell.

Same robot and scene family as the author's sim2cell (imitation learning) and
graspsight (perception + planning) projects — here the task is learned by
reinforcement. Observations are privileged state (like the scripted expert
had); actions are bounded deltas on the position-actuator setpoints.

Tasks:
    ReachEnv — drive the TCP to a random 3D target. Fast to learn; verifies
        the RL plumbing end to end.
    LiftEnv  — grasp the block and raise it. Dense shaping: reach term,
        proximity-gated grasp term, lift term, success bonus.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSETS / "so_arm100" / "pickplace.xml"

BLOCK_HALF = 0.012
JAW_OPEN = 1.4


class WorkCellEnv(gym.Env):
    """Base env: model loading, delta-position actions, stepping, rendering."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    N_ACT = 6                     # 5 arm joints + jaw
    DELTA_MAX = 0.05              # rad per control step on each setpoint

    def __init__(self, max_steps: int = 100, control_hz: float = 15.0, render_mode: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.control_hz = control_hz
        self.n_substeps = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self.render_mode = render_mode
        self._renderer = None
        self._home = self.model.key("home")
        self._block_qadr = self.model.jnt_qposadr[self.model.joint("block_free").id]
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self._t = 0

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.N_ACT,), dtype=np.float32)

    # ------------------------------------------------------------- mechanics
    def _reset_arm(self, noise: float = 0.03) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = self._home.qpos[:6].copy()
        q[:5] += self.np_random.uniform(-noise, noise, size=5)
        self.data.qpos[:6] = q
        self.data.ctrl[:6] = q
        self._t = 0

    def _place_block(self, xy: np.ndarray, yaw: float) -> None:
        adr = self._block_qadr
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], BLOCK_HALF]
        half = 0.5 * yaw
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0, 0, np.sin(half)]

    def _apply_action(self, action: np.ndarray) -> None:
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.data.ctrl[:6] = np.clip(
            self.data.ctrl[:6] + a * self.DELTA_MAX, self._ctrl_lo[:6], self._ctrl_hi[:6]
        )
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._t += 1

    @property
    def tcp(self) -> np.ndarray:
        return self.data.site("tcp").xpos.copy()

    @property
    def block_pos(self) -> np.ndarray:
        return self.data.body("block").xpos.copy()

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=480)
        self._renderer.update_scene(self.data, camera="front")
        return self._renderer.render().copy()


class ReachEnv(WorkCellEnv):
    """Drive the TCP to a random target. Success: within 2 cm."""

    SUCCESS_D = 0.02

    def __init__(self, max_steps: int = 100, **kw):
        super().__init__(max_steps=max_steps, **kw)
        # obs: qpos(5) qvel(5) ctrl(5) tcp(3) target(3) target-tcp(3) = 24
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)

    def _sample_target(self) -> np.ndarray:
        """Sample a target VERIFIED reachable (IK residual < 5 mm) — makes task
        solvability an env property rather than a training-time surprise."""
        from .ik import solve_ik

        for _ in range(50):
            r = self.np_random.uniform(0.18, 0.28)
            th = self.np_random.uniform(np.deg2rad(-30), np.deg2rad(35))
            z = self.np_random.uniform(0.04, 0.15)
            t = np.array([r * np.sin(th), -r * np.cos(th), z])
            _, err = solve_ik(self.model, self.data, t)
            if err < 0.005:
                return t
        raise RuntimeError("could not sample a reachable target")

    def _obs(self) -> np.ndarray:
        d = self.data
        return np.concatenate([
            d.qpos[:5], d.qvel[:5], d.ctrl[:5],
            self.tcp, self.target, self.target - self.tcp,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_arm()
        self._place_block(np.array([0.30, 0.12]), 0.0)   # park the block out of the workspace
        self.target = self._sample_target()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        self._apply_action(action)
        d = float(np.linalg.norm(self.target - self.tcp))
        success = d < self.SUCCESS_D
        reward = -d - 0.005 * float(np.square(action).sum()) + (1.0 if success else 0.0)
        terminated = success
        truncated = self._t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {
            "success": success, "is_success": success, "distance": d,  # is_success: SB3 EvalCallback convention
        }


class LiftEnv(WorkCellEnv):
    """Grasp the block and raise it above 7 cm. Dense shaping."""

    LIFT_Z = 0.07

    def __init__(self, max_steps: int = 150, **kw):
        super().__init__(max_steps=max_steps, **kw)
        # obs: qpos(6) qvel(6) ctrl(6) tcp(3) block(3) block-tcp(3) = 27
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        d = self.data
        return np.concatenate([
            d.qpos[:6], d.qvel[:6], d.ctrl[:6],
            self.tcp, self.block_pos, self.block_pos - self.tcp,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random.uniform(0.19, 0.26)
        th = self.np_random.uniform(np.deg2rad(-20), np.deg2rad(30))
        xy = np.array([r * np.sin(th), -r * np.cos(th)])
        # Yaw aligned with the approach bearing (as in sim2cell's data
        # collection): with the wrist at home roll the faces meet the jaws.
        self._place_block(xy, float(th))
        self.data.ctrl[5] = JAW_OPEN
        self.data.qpos[5] = JAW_OPEN
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        self._apply_action(action)
        b, t = self.block_pos, self.tcp
        d = float(np.linalg.norm(b - t))
        lift = float(np.clip(b[2] - BLOCK_HALF, 0.0, self.LIFT_Z))
        success = bool(b[2] > self.LIFT_Z)

        reward = 0.5 * (1 - np.tanh(10.0 * d))               # reach shaping
        if d < 0.03:                                          # proximity-gated grasp hint
            jaw = float(self.data.qpos[5])
            reward += 0.25 * (JAW_OPEN - jaw) / (JAW_OPEN + 0.174)
        reward += 3.0 * lift / self.LIFT_Z                    # lift shaping
        reward += 5.0 if success else 0.0
        reward -= 0.005 * float(np.square(action).sum())

        terminated = success
        truncated = self._t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {
            "success": success, "is_success": success, "distance": d, "block_z": float(b[2]),
        }
