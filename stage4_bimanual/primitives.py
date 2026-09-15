"""Physics-based scripted primitives for the two SO-101 arms in assets/bimanual_scene_weld.xml.

Grasps are weld-assisted: the arm really moves to the object, the descent stops at first
fingertip contact, and only then is a body weld activated with the relative pose captured at
that instant (so nothing is teleported or snapped). Every skill returns ground-truth metrics
(lift, tilt, gripper/object penetration, drawer travel) and derives `success` from them.

    sim = Sim(); sim.reset(seed=101)
    ok, metrics = pick(sim, sim.arms["B"], "mug")

A frame hook (`sim.frame_hook = lambda sim: ...`) is called every STEPS_PER_FRAME physics steps;
the demonstration recorder uses it to log state/action/image, the review script to render.
"""

from __future__ import annotations

import numpy as np
import mujoco

SCENE = "assets/bimanual_scene_weld.xml"
STEPS_PER_FRAME = 20          # 0.002 s * 20 = 25 fps, matching stage3_policy.learned.schema
TABLE_Z = 0.70
GRIP_OPEN, GRIP_CLOSED = 1.5, 0.05
MAX_JOINT_SPEED = 0.5         # rad/s; the sts3215 servo model (3.35 N.m, kv=2.7) tracks cleanly at this pace
MIN_MOVE_STEPS = 400
TOUCH_MM = 0.8                # guarded moves stop at this penetration instead of pressing on
ROT_WEIGHT = 0.03             # metres of position error traded per radian of approach-axis error
APPROACH_HEIGHT = 0.15
LIFT_HEIGHT = 0.10
MIN_LIFT, MAX_PENETRATION_MM = 0.05, 3.0
DRAWER_OPEN_MIN = 0.12        # of the 0.16 m slide range
HOME_OFFSET = np.array([0.15, 0.0, 0.18])  # hover pose relative to each arm base

# Grasp geometry per object: fingertips seat on the side wall facing the arm, `depth` below the
# top (mug/bottle rim pinch), or descend onto the object until contact (plate rim, utensils).
OBJECTS = {  # targets sit slightly *inside* the object so the guarded descent always ends in contact
    "mug":          dict(radius=0.044, top_z=0.096, depth=0.035, clearance=0.004),
    "water_bottle": dict(radius=0.025, top_z=0.160, depth=0.035, clearance=0.004),
    "plate":        dict(radius=0.050, top_z=0.012, depth=0.006, clearance=0.0, grip=GRIP_CLOSED, side=-1),  # closed fingertips onto the far rim (near rim = arm folds into itself)
    "spoon":        dict(radius=0.0,   top_z=0.010, depth=0.015, clearance=0.0, max_pen=4.0),  # top pinch of the handle
    "fork":         dict(radius=0.0,   top_z=0.010, depth=0.015, clearance=0.0, max_pen=4.0),
}
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class Arm:
    def __init__(self, model: mujoco.MjModel, letter: str):
        self.letter, p = letter, letter.lower()
        self.joint_names = [f"{p}_{j}" for j in JOINTS]
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.qadr = [model.jnt_qposadr[j] for j in jids]
        self.dadr = [model.jnt_dofadr[j] for j in jids]
        self.lo = np.array([model.jnt_range[j][0] for j in jids])
        self.hi = np.array([model.jnt_range[j][1] for j in jids])
        self.ctrl = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.joint_names]
        self.grip_ctrl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{p}_gripper")
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{p}_gripperframe")
        self.gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{p}_gripper_base")
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{p}_base")
        self.geoms = {g for g in range(model.ngeom)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").startswith(f"{p}_")}
        self.home: np.ndarray | None = None

    def q(self, data):
        return np.array([data.qpos[a] for a in self.qadr])

    def set_q(self, data, q):
        for a, v in zip(self.qadr, q):
            data.qpos[a] = v

    def tip(self, data):
        return data.site_xpos[self.site].copy()

    def approach_axis(self, data):
        """The fingertip-frame axis that points most nearly along world z, and its direction."""
        R = data.site_xmat[self.site].reshape(3, 3)
        k = int(np.argmax(np.abs(R[2, :])))
        return k, R[:, k].copy()


class Sim:
    def __init__(self, scene: str = SCENE):
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self.arms = {"A": Arm(self.model, "A"), "B": Arm(self.model, "B")}
        self.frame_hook = None
        self.step_count = 0
        self.table_geom = self.geom("table_surface")
        self.drawer_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        self.tray_body = self.body("sliding_tray")
        self.tray_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.tray_body}
        mujoco.mj_forward(self.model, self.data)
        for arm in self.arms.values():
            hover = self.data.xpos[arm.base_body] + HOME_OFFSET
            arm.home = self.ik(arm, hover, np.array([0.0, -0.9, 1.2, 0.4, 0.0]))

    # ---- ids -------------------------------------------------------------------------------
    def body(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
    def geom(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    def site(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
    def eq(self, name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def object_geoms(self, name):
        b = self.body(name)
        return {g for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == b and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}

    # ---- state -----------------------------------------------------------------------------
    def reset(self, seed: int, jitter: dict[str, float] | None = None):
        """Reset; jitter = {object: max xy offset in m} for initial placement only (never mid-episode)."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.eq_active[:] = 0
        for arm in self.arms.values():
            arm.set_q(self.data, arm.home)
            self.data.ctrl[arm.ctrl] = arm.home
            self.data.ctrl[arm.grip_ctrl] = GRIP_OPEN
        rng = np.random.RandomState(seed)
        for name, amount in (jitter or {}).items():
            adr = self.model.jnt_qposadr[self.model.body_jntadr[self.body(name)]]
            self.data.qpos[adr:adr + 2] += rng.uniform(-amount, amount, size=2)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.settle(100)

    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        if self.frame_hook is not None and self.step_count % STEPS_PER_FRAME == 0:
            self.frame_hook(self)

    def settle(self, steps):
        for _ in range(steps):
            self.step()

    def all_joint_qpos(self, motor_names):
        return np.array([self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
                         for n in motor_names], dtype=np.float32)

    # ---- metrics ---------------------------------------------------------------------------
    def penetration_mm(self, geoms_a, geoms_b):
        worst = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in geoms_a and c.geom2 in geoms_b) or (c.geom2 in geoms_a and c.geom1 in geoms_b):
                worst = max(worst, -c.dist * 1000.0)
        return worst

    def up_axis(self, body):
        """World up expressed in the body frame (bodies are not all z-up: the utensils lie rotated)."""
        return self.data.xmat[body].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])

    def tilt_deg(self, body, up0=None):
        """Angle between the body's initial up direction (up0, from up_axis) and world up."""
        R = self.data.xmat[body].reshape(3, 3)
        up = R @ (up0 if up0 is not None else R.T @ np.array([0.0, 0.0, 1.0]))
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def support_geoms(self):
        return {self.table_geom, self.geom("tray_floor")}

    def resting_on_something(self, obj_geoms, arm: Arm, min_mm=0.2):
        """True once the object touches anything other than the arm holding it (table or another object)."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & obj_geoms and not (pair - obj_geoms) & (arm.geoms | obj_geoms) and -c.dist * 1000.0 > min_mm:
                return True
        return False

    def drawer_opening(self):
        return float(self.data.qpos[self.model.jnt_qposadr[self.drawer_joint]])

    # ---- welds -----------------------------------------------------------------------------
    def weld_on(self, name, body1, body2):
        """Activate weld `name` holding body2 exactly where it is relative to body1 right now."""
        d, m = self.data, self.model
        q1_inv, rel_pos, rel_quat = np.zeros(4), np.zeros(3), np.zeros(4)
        mujoco.mju_negQuat(q1_inv, d.xquat[body1])
        mujoco.mju_rotVecQuat(rel_pos, d.xpos[body2] - d.xpos[body1], q1_inv)
        mujoco.mju_mulQuat(rel_quat, q1_inv, d.xquat[body2])
        e = self.eq(name)
        m.eq_data[e, 0:3] = 0.0
        m.eq_data[e, 3:6] = rel_pos
        m.eq_data[e, 6:10] = rel_quat
        d.eq_active[e] = 1

    def weld_off(self, name):
        self.data.eq_active[self.eq(name)] = 0

    def connect_on(self, name, body1, body2, point):
        """Activate connect (ball-joint) constraint `name` at world `point`: body2 is dragged along
        with body1 but stays free to rotate relative to it (a hooked drawer handle)."""
        d, m = self.data, self.model
        e = self.eq(name)
        for slot, body in ((0, body1), (3, body2)):
            q_inv, local = np.zeros(4), np.zeros(3)
            mujoco.mju_negQuat(q_inv, d.xquat[body])
            mujoco.mju_rotVecQuat(local, np.asarray(point) - d.xpos[body], q_inv)
            m.eq_data[e, slot:slot + 3] = local
        d.eq_active[e] = 1

    def reach_path(self, arm: Arm, start_q, targets, keep_axis=None, tol=0.004):
        """IK waypoints toward successive targets, seeded from the previous solution; stops at the
        first target the arm cannot reach within tol. Returns (waypoints, reached_count)."""
        path, q = [], np.asarray(start_q)
        for target in targets:
            cand = self.ik(arm, target, q, keep_axis=keep_axis)
            s = self._ik_data
            s.qpos[:] = self.data.qpos
            arm.set_q(s, cand)
            mujoco.mj_forward(self.model, s)
            if np.linalg.norm(s.site_xpos[arm.site] - target) > tol:
                break
            path.append(cand)
            q = cand
        return path, len(path)

    # ---- kinematics ------------------------------------------------------------------------
    def ik(self, arm: Arm, target, q_start, iters=300, damping=0.05, keep_axis=None):
        """Damped least-squares IK for the fingertip site on a scratch MjData (the live simulation
        is never teleported). keep_axis=(k, v) also keeps fingertip axis k along v: 2 rotational
        constraints + 3 positional = the arm's 5 DOF, so vertical moves stay vertical."""
        s = self._ik_data
        s.qpos[:] = self.data.qpos
        arm.set_q(s, q_start)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            mujoco.mj_forward(self.model, s)
            err = target - s.site_xpos[arm.site]
            mujoco.mj_jacSite(self.model, s, jacp, jacr, arm.site)
            J = jacp[:, arm.dadr]
            if keep_axis is not None:
                k, v = keep_axis
                axis = s.site_xmat[arm.site].reshape(3, 3)[:, k]
                err = np.concatenate([err, ROT_WEIGHT * np.cross(axis, v)])
                J = np.vstack([J, ROT_WEIGHT * jacr[:, arm.dadr]])
            if np.linalg.norm(err[:3]) < 2e-3 and np.linalg.norm(err[3:]) < 2e-3:
                break
            dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(err)), err), -0.05, 0.05)
            arm.set_q(s, np.clip(arm.q(s) + dq, arm.lo, arm.hi))
        mujoco.mj_forward(self.model, s)
        return arm.q(s).copy()

    def fk_axis(self, arm: Arm, q):
        s = self._ik_data
        s.qpos[:] = self.data.qpos
        arm.set_q(s, q)
        mujoco.mj_forward(self.model, s)
        return arm.approach_axis(s)

    def steps_for(self, q_from, q_to):
        return int(max(MIN_MOVE_STEPS, np.max(np.abs(np.asarray(q_to) - np.asarray(q_from))) / MAX_JOINT_SPEED / self.model.opt.timestep))

    def move(self, arm: Arm, q_to, grip, steps=None, until_touch=None, until=None):
        """Interpolate the arm's joint targets to q_to. The move stops early (holding the current
        pose) at the first arm contact with the until_touch geoms, or when until(sim) is true.
        Returns the joint target actually held at the end."""
        q_from = np.array(self.data.ctrl[arm.ctrl])
        steps = steps or self.steps_for(q_from, q_to)
        target = q_to
        for i in range(steps):
            target = q_from + (i + 1) / steps * (q_to - q_from)
            self.data.ctrl[arm.ctrl] = target
            self.data.ctrl[arm.grip_ctrl] = grip
            self.step()
            touched = until_touch is not None and self.penetration_mm(arm.geoms, until_touch) > TOUCH_MM
            if touched or (until is not None and until(self)):
                target = arm.q(self.data)
                self.data.ctrl[arm.ctrl] = target
                break
        return target

    def move_line(self, arm: Arm, target, grip, keep_axis=None, spacing=0.02):
        """Straight fingertip line to target as IK waypoints (keeps the approach axis steady on
        long carries instead of one big joint-space jump)."""
        start = arm.tip(self.data)
        n = max(1, int(np.ceil(np.linalg.norm(target - start) / spacing)))
        targets = [start + (target - start) * (i / n) for i in range(1, n + 1)]
        path, reached = self.reach_path(arm, arm.q(self.data), targets, keep_axis=keep_axis)
        self.move_path(arm, path, grip)
        return reached == n

    def align_yaw(self, arm: Arm, obj: str, grip, desired=np.array([1.0, 0.0])):
        """Rotate the wrist roll so the held object's long axis (its collision box z) lies along
        `desired` in the table plane; utensils are laid down parallel to the table edge."""
        g = self.geom(f"{obj}_geom")
        long_axis = self.data.geom_xmat[g].reshape(3, 3)[:, 2]
        err = np.arctan2(long_axis[1], long_axis[0]) - np.arctan2(desired[1], desired[0])
        err = (err + np.pi / 2) % np.pi - np.pi / 2                    # the long axis has no front/back
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, jacr, arm.site)
        dyaw_droll = jacr[2, arm.dadr[4]]                                 # world yaw per unit wrist roll
        if abs(dyaw_droll) < 0.3:
            return False
        q = np.array(self.data.ctrl[arm.ctrl])
        q[4] = np.clip(q[4] - err / dyaw_droll, arm.lo[4], arm.hi[4])
        self.move(arm, q, grip)
        return True

    def hold(self, arm: Arm, grip, steps):
        self.move(arm, np.array(self.data.ctrl[arm.ctrl]), grip, steps=steps)

    def move_path(self, arm: Arm, waypoints, grip, min_steps=40):
        for q in waypoints:
            steps = max(min_steps, self.steps_for(self.data.ctrl[arm.ctrl], q))
            self.move(arm, q, grip, steps=steps)


# ---- skills ------------------------------------------------------------------------------------
def weld_name(arm: Arm, obj: str):
    return f"{arm.letter.lower()}_grasp_{obj}"


def grasp_point(sim: Sim, arm: Arm, obj: str):
    spec = OBJECTS[obj]
    pos = sim.data.xpos[sim.body(obj)].copy()
    pos[:2] = sim.data.geom_xpos[sim.geom(f"{obj}_geom")][:2]   # the collision geom, not the (offset) body origin
    to_arm = sim.data.xpos[arm.base_body][:2] - pos[:2]
    to_arm /= np.linalg.norm(to_arm)
    side = np.array([*(spec.get("side", 1) * to_arm * (spec["radius"] + spec["clearance"])), 0.0])
    return pos + side + np.array([0.0, 0.0, spec["top_z"] - spec["depth"]])


def pick(sim: Sim, arm: Arm, obj: str):
    """Approach from above, guarded descent to the object, close, weld, lift LIFT_HEIGHT."""
    body = sim.body(obj)
    obj_geoms = sim.object_geoms(obj)
    z0 = sim.data.xpos[body][2]
    up0 = sim.up_axis(body)
    metrics = {"max_tilt_deg": 0.0, "max_penetration_mm": 0.0}

    def track():
        metrics["max_tilt_deg"] = max(metrics["max_tilt_deg"], sim.tilt_deg(body, up0))
        metrics["max_penetration_mm"] = max(metrics["max_penetration_mm"], sim.penetration_mm(arm.geoms, obj_geoms))
    prev_hook = sim.frame_hook
    sim.frame_hook = lambda s: (track(), prev_hook(s) if prev_hook else None)
    try:
        grasp = grasp_point(sim, arm, obj)
        above = grasp + np.array([0.0, 0.0, APPROACH_HEIGHT])
        grip = OBJECTS[obj].get("grip", GRIP_OPEN)
        q_above = sim.ik(arm, above, arm.q(sim.data))
        axis = sim.fk_axis(arm, q_above)
        q_grasp = sim.ik(arm, grasp, q_above, keep_axis=axis)
        sim.move(arm, q_above, grip)
        sim.move(arm, q_grasp, grip, until_touch=obj_geoms | sim.support_geoms())
        metrics["touched_object"] = float(sim.penetration_mm(arm.geoms, obj_geoms) > 0.0)
        sim.hold(arm, GRIP_CLOSED, 200)
        sim.weld_on(weld_name(arm, obj), arm.gripper_body, body)
        q_lift = sim.ik(arm, arm.tip(sim.data) + np.array([0.0, 0.0, LIFT_HEIGHT]), arm.q(sim.data), keep_axis=arm.approach_axis(sim.data))
        sim.move(arm, q_lift, GRIP_CLOSED)
    finally:
        sim.frame_hook = prev_hook
    metrics["lift_m"] = float(sim.data.xpos[body][2] - z0)
    success = (metrics["touched_object"] == 1.0 and metrics["lift_m"] > MIN_LIFT
               and metrics["max_penetration_mm"] < OBJECTS[obj].get("max_pen", MAX_PENETRATION_MM))
    return bool(success), metrics


def place(sim: Sim, arm: Arm, obj: str, xy, align=True):
    """Carry a welded object over xy, lower it until it rests on something, release, and retract.
    align: lay utensils parallel to the table edge (skip for hand-offs, where it only adds motion)."""
    body = sim.body(obj)
    obj_geoms = sim.object_geoms(obj)
    up0 = sim.up_axis(body)
    metrics = {"max_tilt_deg": 0.0}
    prev_hook = sim.frame_hook
    sim.frame_hook = lambda s: (metrics.__setitem__("max_tilt_deg", max(metrics["max_tilt_deg"], sim.tilt_deg(body, up0))), prev_hook(s) if prev_hook else None)
    try:
        axis = arm.approach_axis(sim.data)
        carry_z = arm.tip(sim.data)[2]
        def over_target():
            # fingertip position that puts the object's origin over xy; the fingertip-to-object offset
            # (rim grasps are ~5 cm off-centre) rotates with the arm's yaw, so it is re-measured after each move
            offset = arm.tip(sim.data) - sim.data.xpos[body]
            return np.array([xy[0], xy[1], carry_z - offset[2]]) + offset
        metrics["carry_reached"] = float(sim.move_line(arm, over_target(), GRIP_CLOSED, keep_axis=axis))
        if align and OBJECTS[obj]["radius"] == 0.0:                   # utensil: lay it parallel to the table edge
            metrics["yaw_aligned"] = float(sim.align_yaw(arm, obj, GRIP_CLOSED))
        for _ in range(2):                                            # corrective moves for the rotated offset
            if np.linalg.norm(sim.data.xpos[body][:2] - np.asarray(xy)) < 0.005:
                break
            sim.move_line(arm, over_target(), GRIP_CLOSED, keep_axis=axis)
        over = over_target()
        offset = arm.tip(sim.data) - sim.data.xpos[body]
        down = over.copy()
        down[2] = TABLE_Z + offset[2] - 0.02                          # aim below the surface; the contact check stops it
        q_down = sim.ik(arm, down, arm.q(sim.data), keep_axis=axis)
        sim.move(arm, q_down, GRIP_CLOSED, until=lambda s: s.resting_on_something(obj_geoms, arm))
        sim.weld_off(weld_name(arm, obj))
        sim.hold(arm, GRIP_OPEN, 200)
        up = arm.tip(sim.data) + np.array([0.0, 0.0, APPROACH_HEIGHT])
        sim.move(arm, sim.ik(arm, up, arm.q(sim.data), keep_axis=axis), GRIP_OPEN)
        sim.settle(100)
    finally:
        sim.frame_hook = prev_hook
    pos = sim.data.xpos[body]
    metrics["xy_error_m"] = float(np.linalg.norm(pos[:2] - np.asarray(xy)))
    metrics["final_tilt_deg"] = sim.tilt_deg(body, up0)
    metrics["height_above_table_m"] = float(pos[2] - TABLE_Z)
    success = metrics["xy_error_m"] < 0.03 and metrics["height_above_table_m"] < 0.03 and metrics["final_tilt_deg"] < 10
    return bool(success), metrics


def open_drawer(sim: Sim, arm: Arm, pull=0.16):
    """Hook the D-handle from above (fingertip in the gap behind the bar), attach a ball-joint
    constraint at the fingertip, pull the slide open along -x as far as the arm can reach
    (wrist free to rotate), release and retract."""
    hook_eq = f"{arm.letter.lower()}_hook_drawer"
    handle = sim.data.site_xpos[sim.site("drawer_handle_site")].copy()
    hook = handle + np.array([0.010, 0.0, -0.004])                # behind the 7 mm bar, just below its centre
    above = hook + np.array([0.0, 0.0, 0.12])
    metrics = {"max_penetration_mm": 0.0}
    prev_hook = sim.frame_hook
    sim.frame_hook = lambda s: (metrics.__setitem__("max_penetration_mm", max(metrics["max_penetration_mm"], sim.penetration_mm(arm.geoms, sim.tray_geoms))), prev_hook(s) if prev_hook else None)
    try:
        q_above = sim.ik(arm, above, arm.q(sim.data))
        axis = sim.fk_axis(arm, q_above)
        sim.move(arm, q_above, GRIP_CLOSED)
        q_hook = sim.ik(arm, hook, q_above, keep_axis=axis)
        sim.move(arm, q_hook, GRIP_CLOSED, until_touch=sim.tray_geoms)
        sim.hold(arm, GRIP_CLOSED, 100)
        tip = arm.tip(sim.data)
        sim.connect_on(hook_eq, arm.gripper_body, sim.tray_body, tip)
        targets = [tip + np.array([-0.01 * i, 0.0, 0.0]) for i in range(1, int(round(pull / 0.01)) + 1)]
        path, n = sim.reach_path(arm, arm.q(sim.data), targets)
        metrics["pull_reachable_m"] = 0.01 * n
        sim.move_path(arm, path, GRIP_CLOSED)
        sim.hold(arm, GRIP_CLOSED, 100)
        metrics["opening_m"] = sim.drawer_opening()
        sim.weld_off(hook_eq)
        up = arm.tip(sim.data) + np.array([0.0, 0.0, 0.12])
        sim.move(arm, sim.ik(arm, up, arm.q(sim.data)), GRIP_OPEN)
        sim.settle(100)
    finally:
        sim.frame_hook = prev_hook
    metrics["final_opening_m"] = sim.drawer_opening()
    success = metrics["final_opening_m"] > DRAWER_OPEN_MIN and metrics["max_penetration_mm"] < MAX_PENETRATION_MM
    return bool(success), metrics


def go_home(sim: Sim, arm: Arm):
    sim.move(arm, arm.home, GRIP_OPEN)
