"""Damped least-squares IK for the SO-ARM100, with commandable jaw yaw.

Adapted from the author's sim2cell project. Same gripper-down constraint
(q_pitch + q_elbow + q_wrist_pitch = 1.57 keeps the last link vertical), with
one addition for grasping at arbitrary object yaw: when the gripper points
down, Wrist_Roll rotates the jaws about the vertical axis, so it is pinned to
a caller-supplied value (computed by the grasp planner) and the remaining
four joints solve the 3 position + 1 downness constraints — a fully
determined system.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
_SCRATCH_CACHE: dict[int, "mujoco.MjData"] = {}
DOWN_SUM = 1.57      # q_pitch + q_elbow + q_wrist_pitch for a downward gripper
HOME_ROLL = -1.57    # Wrist_Roll at the model's home keyframe


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    q_init: np.ndarray | None = None,
    wrist_roll: float | None = None,
    site: str = "tcp",
    max_iters: int = 120,
    tol: float = 1e-4,
    damping: float = 1e-3,
    step_clip: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Position IK with gripper-down posture and optional pinned jaw yaw.

    Args:
        target_pos: world xyz for the TCP site.
        q_init: 5-dim starting guess (defaults to current qpos).
        wrist_roll: if given, Wrist_Roll is held at this value and excluded
            from the least-squares solve; if None, it participates (sim2cell
            behavior).

    Returns:
        (q, err): 5-dim arm joint solution and final position error [m].
    """
    site_id = model.site(site).id
    jids = [model.joint(j).id for j in ARM_JOINTS]
    qadrs = [model.jnt_qposadr[jid] for jid in jids]
    dadrs = [model.jnt_dofadr[jid] for jid in jids]
    ranges = np.array([model.jnt_range[jid] for jid in jids])

    # Reuse one scratch MjData per model: allocating a fresh one per call
    # exhausted memory under a 50-calls-per-reset validation loop.
    key = id(model)
    scratch = _SCRATCH_CACHE.get(key)
    if scratch is None:
        scratch = mujoco.MjData(model)
        _SCRATCH_CACHE[key] = scratch
    scratch.qpos[:] = data.qpos
    q = np.array([data.qpos[a] for a in qadrs]) if q_init is None else np.asarray(q_init, dtype=float).copy()

    if wrist_roll is not None:
        q[4] = float(np.clip(wrist_roll, ranges[4, 0], ranges[4, 1]))
        free = [0, 1, 2, 3]          # joints in the solve
    else:
        free = [0, 1, 2, 3, 4]

    target_pos = np.asarray(target_pos, dtype=float)
    err = np.inf
    for _ in range(max_iters):
        for a, qi in zip(qadrs, q):
            scratch.qpos[a] = qi
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

        e_pos = target_pos - scratch.site_xpos[site_id]
        e_down = DOWN_SUM - (q[1] + q[2] + q[3])
        err = float(np.linalg.norm(e_pos))
        if err < tol and abs(e_down) < 1e-3:
            break

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
        J_pos = jacp[:, [dadrs[i] for i in free]]                  # (3, n_free)
        J_down = np.array([[1.0 if i in (1, 2, 3) else 0.0 for i in free]])

        J = np.vstack([J_pos, J_down])
        e = np.concatenate([e_pos, [e_down]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(4), e)
        dq = np.clip(dq, -step_clip, step_clip)
        for k, i in enumerate(free):
            q[i] = np.clip(q[i] + dq[k], ranges[i, 0], ranges[i, 1])

    return q, err
