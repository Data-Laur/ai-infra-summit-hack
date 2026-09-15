import os
import json
import mujoco
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from stage3_policy.learned import schema

FPS = schema.FPS
PHYSICS_STEPS_PER_FRAME = schema.PHYSICS_STEPS_PER_FRAME
MOTOR_NAMES = schema.MOTOR_NAMES
STATE_KEY, ACTION_KEY = schema.STATE_KEY, schema.ACTION_KEY
IMAGE_KEY = schema.IMAGE_KEY
SUCCESS_KEY, DONE_KEY = schema.SUCCESS_KEY, schema.DONE_KEY
CODEBASE_VERSION, ROBOT_TYPE = schema.CODEBASE_VERSION, schema.ROBOT_TYPE
INSTRUCTION = "pick up the mug with arm B"
REPO_ID = "local/butler_demos_pick"  # local-only label; never resolved against the HF Hub
EVAL_SEEDS = set(range(10))

B_JOINTS = ["b_shoulder_pan","b_shoulder_lift","b_elbow_flex","b_wrist_flex","b_wrist_roll"]
LIMITS_B = {"b_shoulder_pan": (-1.91986,1.91986), "b_shoulder_lift": (-1.74533,1.74533),
            "b_elbow_flex": (-1.69,1.69), "b_wrist_flex": (-1.65806,1.65806),
            "b_wrist_roll": (-2.74385,2.84121)}

# mug_geom: cylinder radius 0.044, half-height 0.048 at z+0.048 in the mug body -> wall at r=0.044, rim at z+0.096
MUG_RADIUS, MUG_RIM_Z = 0.044, 0.096
GRASP_DEPTH_BELOW_RIM = 0.035   # fingertip frame this far below the rim, on the wall facing the arm
GRASP_WALL_CLEARANCE = 0.004    # fingertips seat just outside the wall instead of pressing into it
APPROACH_HEIGHT = 0.15
LIFT_HEIGHT = 0.10
MIN_LIFT, MAX_PENETRATION_MM = 0.05, 3.0   # success: lifted, and the mug never sank into the gripper
MAX_JOINT_SPEED = 0.5   # rad/s; the sts3215 servo model (3.35 N.m, kv=2.7) tracks cleanly at this pace
MIN_MOVE_STEPS = 400
TOUCH_MM = 0.8          # descent stops at first fingertip contact instead of pressing on the rim

OUT_ROOT = Path("data/butler_demos/pick")

model = mujoco.MjModel.from_xml_path("assets/bimanual_scene_weld.xml")

jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in B_JOINTS]
qpos_addrs = [model.jnt_qposadr[j] for j in jids]
dof_addrs  = [model.jnt_dofadr[j]  for j in jids]
lo = np.array([LIMITS_B[n][0] for n in B_JOINTS]); hi = np.array([LIMITS_B[n][1] for n in B_JOINTS])
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "b_gripperframe")
gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b_gripper_base")
mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
mug_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mug_geom")
mug_joint = model.body_jntadr[mug_body]  # freejoint has no name; get via the body's joint address
mug_qpos_addr = model.jnt_qposadr[mug_joint]
eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "b_grasp_mug")
ALL_MOTOR_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in MOTOR_NAMES]
ALL_QPOS = [model.jnt_qposadr[j] for j in ALL_MOTOR_IDS]
ARM_B_GEOMS = {g for g in range(model.ngeom)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").startswith("b_")}

def get_q(data): return np.array([data.qpos[a] for a in qpos_addrs])
def set_q(data, q):
    for addr, val in zip(qpos_addrs, q): data.qpos[addr] = val

_ik_data = mujoco.MjData(model)
ROT_WEIGHT = 0.03  # metres of position error traded per radian of orientation error

def ik_to(model, data, target, q_start, iters=300, damping=0.05, keep_axis=None):
    """Damped least-squares IK for the arm-B fingertip site, solved on a scratch MjData so the live
    simulation is never teleported. keep_axis=(k, v) additionally keeps site axis k pointing along v
    (2 rotational constraints: with 3 for position that matches the arm's 5 DOF, yaw stays free),
    so descents and lifts stay vertical instead of pitching the wrist and the held mug."""
    _ik_data.qpos[:] = data.qpos
    set_q(_ik_data, q_start)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for it in range(iters):
        mujoco.mj_forward(model, _ik_data)
        err = target - _ik_data.site_xpos[site_id]
        mujoco.mj_jacSite(model, _ik_data, jacp, jacr, site_id)
        J = jacp[:, dof_addrs]
        if keep_axis is not None:
            k, v = keep_axis
            axis = _ik_data.site_xmat[site_id].reshape(3, 3)[:, k]
            err = np.concatenate([err, ROT_WEIGHT * np.cross(axis, v)])
            J = np.vstack([J, ROT_WEIGHT * jacr[:, dof_addrs]])
        if np.linalg.norm(err[:3]) < 2e-3 and np.linalg.norm(err[3:]) < 2e-3: break
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping**2*np.eye(len(err)), err), -0.05, 0.05)
        set_q(_ik_data, np.clip(get_q(_ik_data)+dq, lo, hi))
    mujoco.mj_forward(model, _ik_data)
    return get_q(_ik_data).copy()

def approach_axis(data):
    """The site axis that points most nearly along world z, and its current direction."""
    R = data.site_xmat[site_id].reshape(3, 3)
    k = int(np.argmax(np.abs(R[2, :])))
    return k, R[:, k].copy()

def full_state(data):
    return np.array([data.qpos[a] for a in ALL_QPOS], dtype=np.float32)

def grasp_point(mug_pos):
    """Fingertip target on the mug wall facing arm B (which sits at -x), just below the rim."""
    return mug_pos + np.array([-(MUG_RADIUS + GRASP_WALL_CLEARANCE), 0.0, MUG_RIM_Z - GRASP_DEPTH_BELOW_RIM])

def steps_for(q_from, q_to):
    """Physics steps so that no joint has to exceed MAX_JOINT_SPEED."""
    return int(max(MIN_MOVE_STEPS, np.max(np.abs(q_to - q_from)) / MAX_JOINT_SPEED / model.opt.timestep))

def mug_tilt_deg(data):
    return float(np.degrees(np.arccos(np.clip(data.xmat[mug_body].reshape(3, 3)[2, 2], -1.0, 1.0))))

def mug_penetration_mm(data):
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {c.geom1, c.geom2}
        if mug_geom in pair and pair & ARM_B_GEOMS:
            worst = max(worst, -c.dist * 1000.0)
    return worst

def weld_mug_to_gripper(data):
    """Activate the weld holding the mug exactly where it is relative to the gripper body."""
    q_grip_inv, rel_pos, rel_quat = np.zeros(4), np.zeros(3), np.zeros(4)
    mujoco.mju_negQuat(q_grip_inv, data.xquat[gripper_body])
    mujoco.mju_rotVecQuat(rel_pos, data.xpos[mug_body] - data.xpos[gripper_body], q_grip_inv)
    mujoco.mju_mulQuat(rel_quat, q_grip_inv, data.xquat[mug_body])
    model.eq_data[eq_id, 0:3] = 0.0          # anchor at the mug origin
    model.eq_data[eq_id, 3:6] = rel_pos      # relpose: mug in gripper-body frame
    model.eq_data[eq_id, 6:10] = rel_quat
    data.eq_active[eq_id] = 1

_renderer = mujoco.Renderer(model, height=480, width=640)

# Home = hovering above the nominal grasp point, so the arm never starts in contact with the mug.
_nominal = mujoco.MjData(model)
mujoco.mj_forward(model, _nominal)
HOME = ik_to(model, _nominal, grasp_point(_nominal.xpos[mug_body].copy()) + np.array([0, 0, APPROACH_HEIGHT]),
             np.array([0.0, -0.9, 1.2, 0.4, 0.0]))

def record_episode(seed, split, on_frame=None):
    data = mujoco.MjData(model)
    rng = np.random.RandomState(seed)
    mujoco.mj_resetData(model, data)
    data.eq_active[eq_id] = 0
    set_q(data, HOME)
    # domain-randomize mug xy position slightly (physics_only-safe: this is INITIAL placement, not teleport-during-episode)
    dx, dy = rng.uniform(-0.02,0.02), rng.uniform(-0.02,0.02)
    data.qpos[mug_qpos_addr] += dx
    data.qpos[mug_qpos_addr+1] += dy
    mujoco.mj_forward(model, data)

    frames = []
    metrics = {"max_tilt_deg": 0.0, "max_penetration_mm": 0.0}

    def log_frame():
        _renderer.update_scene(data, camera="overhead_cam")
        img = _renderer.render()
        frames.append({
            "state": full_state(data),
            "action": np.array(data.ctrl, dtype=np.float32).copy(),
            "image": img,
        })
        metrics["max_tilt_deg"] = max(metrics["max_tilt_deg"], mug_tilt_deg(data))
        metrics["max_penetration_mm"] = max(metrics["max_penetration_mm"], mug_penetration_mm(data))
        if on_frame is not None:
            on_frame(data, len(frames) - 1)

    def move_smooth(q_from, q_to, gripper_val, steps, until_touch=False):
        """Interpolate the arm target; with until_touch the move stops (holding the current pose)
        at the first fingertip contact with the mug, so the descent seats the fingers instead of
        pressing them into the rim. Returns the joint target actually held at the end."""
        target = q_to
        for i in range(steps):
            alpha = (i+1)/steps
            target = q_from + alpha*(q_to - q_from)
            data.ctrl[6:11] = target
            data.ctrl[11] = gripper_val
            mujoco.mj_step(model, data)
            if until_touch and mug_penetration_mm(data) > TOUCH_MM:
                target = get_q(data)
                data.ctrl[6:11] = target
                break
            if (i+1) % PHYSICS_STEPS_PER_FRAME == 0:
                log_frame()
        return target

    mug_pos0 = data.xpos[mug_body].copy()
    grasp = grasp_point(mug_pos0)
    above = grasp + np.array([0, 0, APPROACH_HEIGHT])

    GRIP_OPEN, GRIP_CLOSED = 1.5, 0.05
    q_above = ik_to(model, data, above, HOME)
    set_q(_ik_data, q_above); mujoco.mj_forward(model, _ik_data)
    q_grasp = ik_to(model, data, grasp, q_above, keep_axis=approach_axis(_ik_data))

    move_smooth(HOME, HOME, GRIP_OPEN, 100)
    move_smooth(HOME, q_above, GRIP_OPEN, steps_for(HOME, q_above))
    q_touch = move_smooth(q_above, q_grasp, GRIP_OPEN, steps_for(q_above, q_grasp), until_touch=True)
    move_smooth(q_touch, q_touch, GRIP_CLOSED, 200)
    weld_mug_to_gripper(data)
    q_now = get_q(data)
    q_lift = ik_to(model, data, data.site_xpos[site_id] + np.array([0, 0, LIFT_HEIGHT]), q_now, keep_axis=approach_axis(data))
    move_smooth(q_now, q_lift, GRIP_CLOSED, steps_for(q_now, q_lift))

    metrics["lift_m"] = float(data.xpos[mug_body][2] - mug_pos0[2])
    success = bool(metrics["lift_m"] > MIN_LIFT and metrics["max_penetration_mm"] < MAX_PENETRATION_MM)
    return frames, success, metrics


def get_or_create_dataset():
    """Real LeRobotDataset writer: guarantees the on-disk layout matches what
    lerobot-train actually loads (chunked parquet, meta/episodes/, meta/tasks.jsonl,
    PNG images), instead of a hand-rolled approximation."""
    if (OUT_ROOT / "meta" / "info.json").exists():
        return LeRobotDataset.resume(repo_id=REPO_ID, root=str(OUT_ROOT))
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=str(OUT_ROOT),
        robot_type=ROBOT_TYPE,
        features=schema.lerobot_features(use_videos=False),
        use_videos=False,
    )


def main():
    seeds = [int(s) for s in os.environ.get("SEEDS", "101").split(",")]
    split = os.environ.get("SPLIT", "train")

    dataset = get_or_create_dataset()

    meta_path = OUT_ROOT / "meta" / "butler_episodes.json"
    episodes_meta = json.loads(meta_path.read_text())["episodes"] if meta_path.exists() else []

    for seed in seeds:
        episode_index = dataset.meta.total_episodes
        frames, success, metrics = record_episode(seed, split)
        n = len(frames)
        for fi, fr in enumerate(frames):
            dataset.add_frame({
                STATE_KEY: fr["state"],
                ACTION_KEY: fr["action"],
                IMAGE_KEY: fr["image"],
                SUCCESS_KEY: np.array([success and fi == n - 1]),
                DONE_KEY: np.array([fi == n - 1]),
                "task": INSTRUCTION,
            })
        dataset.save_episode()
        episodes_meta.append({
            "episode_index": episode_index, "seed": seed, "skill": "pick", "arm": "B",
            "object": "mug", "instruction": INSTRUCTION, "split": split,
            "success": success, "physics_only": True,
            "lift_m": round(metrics["lift_m"], 4),
            "max_tilt_deg": round(metrics["max_tilt_deg"], 1),
            "max_penetration_mm": round(metrics["max_penetration_mm"], 2),
        })
        print(f"episode {episode_index} seed={seed} split={split} frames={n} success={success} "
              f"lift={metrics['lift_m']:.3f}m tilt={metrics['max_tilt_deg']:.1f}deg "
              f"penetration={metrics['max_penetration_mm']:.1f}mm")

    dataset.finalize()
    meta_path.write_text(json.dumps({"episodes": episodes_meta}, indent=2))
    print("dataset now has", dataset.meta.total_episodes, "episodes,", dataset.meta.total_frames, "frames at", OUT_ROOT)


if __name__ == "__main__":
    main()
