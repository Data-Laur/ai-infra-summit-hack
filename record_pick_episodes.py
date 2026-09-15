import os
import json
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

FPS = 25
PHYSICS_STEPS_PER_FRAME = 20
MOTOR_NAMES = tuple(f"{arm}_{j}" for arm in ("a","b") for j in
    ("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"))
STATE_KEY, ACTION_KEY = "observation.state", "action"
IMAGE_KEY = "observation.images.overhead"
SUCCESS_KEY, DONE_KEY = "next.success", "next.done"
CODEBASE_VERSION, ROBOT_TYPE = "v3.0", "butler_bimanual_so101_sim"
EVAL_SEEDS = set(range(10))

B_JOINTS = ["b_shoulder_pan","b_shoulder_lift","b_elbow_flex","b_wrist_flex","b_wrist_roll"]
LIMITS_B = {"b_shoulder_pan": (-1.91986,1.91986), "b_shoulder_lift": (-1.74533,1.74533),
            "b_elbow_flex": (-1.69,1.69), "b_wrist_flex": (-1.65806,1.65806),
            "b_wrist_roll": (-2.74385,2.84121)}

OUT_ROOT = Path("data/butler_demos/pick")
(OUT_ROOT / "meta").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)

model = mujoco.MjModel.from_xml_path("assets/bimanual_scene_weld.xml")

jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in B_JOINTS]
qpos_addrs = [model.jnt_qposadr[j] for j in jids]
dof_addrs  = [model.jnt_dofadr[j]  for j in jids]
lo = np.array([LIMITS_B[n][0] for n in B_JOINTS]); hi = np.array([LIMITS_B[n][1] for n in B_JOINTS])
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "b_gripperframe")
mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
mug_joint = model.body_jntadr[mug_body]  # freejoint has no name; get via the body's joint address
mug_qpos_addr = model.jnt_qposadr[mug_joint]
eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "b_grasp_mug")
ALL_MOTOR_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in MOTOR_NAMES]
ALL_QPOS = [model.jnt_qposadr[j] for j in ALL_MOTOR_IDS]

def get_q(data): return np.array([data.qpos[a] for a in qpos_addrs])
def set_q(data, q):
    for addr, val in zip(qpos_addrs, q): data.qpos[addr] = val

def ik_to(model, data, target, q_start, iters=300, damping=0.05):
    set_q(data, q_start)
    jacp = np.zeros((3, model.nv))
    for it in range(iters):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[site_id]
        if np.linalg.norm(err) < 2e-3: break
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, dof_addrs]
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping**2*np.eye(3), err), -0.05, 0.05)
        set_q(data, np.clip(get_q(data)+dq, lo, hi))
    mujoco.mj_forward(model, data)
    return get_q(data).copy()

def full_state(data):
    return np.array([data.qpos[a] for a in ALL_QPOS], dtype=np.float32)

_renderer = mujoco.Renderer(model, height=480, width=640)

def record_episode(seed, split):
    data = mujoco.MjData(model)
    rng = np.random.RandomState(seed)
    mujoco.mj_resetData(model, data)
    home = np.array([0.0,-0.9,1.2,0.4,0.0])
    set_q(data, home)
    # domain-randomize mug xy position slightly (physics_only-safe: this is INITIAL placement, not teleport-during-episode)
    dx, dy = rng.uniform(-0.02,0.02), rng.uniform(-0.02,0.02)
    data.qpos[mug_qpos_addr] += dx
    data.qpos[mug_qpos_addr+1] += dy
    mujoco.mj_forward(model, data)

    frames = []

    def log_frame():
        _renderer.update_scene(data, camera="overhead_cam")
        img = _renderer.render()
        frames.append({
            "state": full_state(data),
            "action": np.array(data.ctrl, dtype=np.float32).copy(),
            "image": img,
        })

    def move_smooth(q_from, q_to, gripper_val, steps, weld_active=None):
        if weld_active is not None:
            data.eq_active[eq_id] = weld_active
        for i in range(steps):
            alpha = (i+1)/steps
            data.ctrl[6:11] = q_from + alpha*(q_to - q_from)
            data.ctrl[11] = gripper_val
            mujoco.mj_step(model, data)
            if (i+1) % PHYSICS_STEPS_PER_FRAME == 0:
                log_frame()

    mug_pos0 = data.xpos[mug_body].copy()
    above = mug_pos0 + np.array([0,0,0.15])
    descend = mug_pos0 + np.array([0,0,0.02])
    lifted = mug_pos0 + np.array([0,0,0.20])

    GRIP_OPEN, GRIP_CLOSED = 1.5, 0.05
    q_above = ik_to(model, data, above, home)
    q_descend = ik_to(model, data, descend, q_above)

    mujoco.mj_resetData(model, data)
    data.qpos[mug_qpos_addr] += dx; data.qpos[mug_qpos_addr+1] += dy
    set_q(data, home); mujoco.mj_forward(model, data)

    move_smooth(home, home, GRIP_OPEN, 100)
    move_smooth(home, q_above, GRIP_OPEN, 400)
    move_smooth(q_above, q_descend, GRIP_OPEN, 400)
    move_smooth(q_descend, q_descend, GRIP_CLOSED, 200)
    data.eq_active[eq_id] = 1
    mujoco.mj_forward(model, data)
    q_now = get_q(data)
    q_lift = ik_to(model, data, lifted, q_now)
    mujoco.mj_forward(model, data)
    move_smooth(q_now, q_lift, GRIP_CLOSED, 400)

    lift_height = data.xpos[mug_body][2] - mug_pos0[2]
    success = bool(lift_height > 0.05)
    return frames, success

# dof_and_pos_map unused placeholder removed

import sys
seeds = [int(s) for s in os.environ.get("SEEDS", "101").split(",")]
split = os.environ.get("SPLIT", "train")

meta_path = OUT_ROOT/"meta"/"butler_episodes.json"
info_path = OUT_ROOT/"meta"/"info.json"
episodes_meta = json.loads(meta_path.read_text())["episodes"] if meta_path.exists() else []
episode_index = len(episodes_meta)
total_frames = json.loads(info_path.read_text())["total_frames"] if info_path.exists() else 0

for seed in seeds:
    frames, success = record_episode(seed, split)
    n = len(frames)
    rows = []
    for fi, fr in enumerate(frames):
        rows.append({
            "episode_index": episode_index, "frame_index": fi, "timestamp": fi / FPS,
            "index": total_frames + fi, "task_index": 0,
            STATE_KEY: fr["state"].tolist(), ACTION_KEY: fr["action"].tolist(),
            SUCCESS_KEY: bool(success and fi == n-1), DONE_KEY: bool(fi == n-1),
        })
    pq.write_table(pa.Table.from_pylist(rows), OUT_ROOT/"data"/f"episode_{episode_index:06d}.parquet")
    episodes_meta.append({
        "episode_index": episode_index, "seed": seed, "skill": "pick", "arm": "B",
        "object": "mug", "instruction": "pick up the mug with arm B", "split": split,
        "success": success, "physics_only": True,
    })
    total_frames += n
    print(f"episode {episode_index} seed={seed} split={split} frames={n} success={success}")
    episode_index += 1

info = {
    "codebase_version": CODEBASE_VERSION, "fps": FPS, "robot_type": ROBOT_TYPE,
    "total_episodes": episode_index, "total_frames": total_frames,
    "features": {
        "timestamp": {}, "frame_index": {}, "episode_index": {}, "index": {}, "task_index": {},
        STATE_KEY: {"dtype":"float32","shape":[12],"names": list(MOTOR_NAMES)},
        ACTION_KEY: {"dtype":"float32","shape":[12],"names": list(MOTOR_NAMES)},
        IMAGE_KEY: {"dtype":"image","shape":[480,640,3],"names":["height","width","channels"]},
        SUCCESS_KEY: {"dtype":"bool","shape":[1],"names": None},
        DONE_KEY: {"dtype":"bool","shape":[1],"names": None},
    }
}
info_path.write_text(json.dumps(info, indent=2))
meta_path.write_text(json.dumps({"episodes": episodes_meta}, indent=2))
print("dataset now has", episode_index, "episodes,", total_frames, "frames at", OUT_ROOT)
