"""Run a sequence of primitives with a free camera and print their ground-truth metrics.

    python -m stage4_bimanual.review_skill pick:B:mug
    python -m stage4_bimanual.review_skill open_drawer:A pick:A:spoon place:A:spoon:0.06:-0.05
    python -m stage4_bimanual.review_skill --seed 201 pick:B:mug

Each argument is skill:arm[:object[:x:y]]. Writes outputs/review/<name>_closeups.png (6 key
frames of the whole sequence) and <name>_slow.mp4, where <name> joins the skills.
"""

from __future__ import annotations

import sys
from pathlib import Path

import av
import cv2
import mujoco
import numpy as np
from PIL import Image

from stage4_bimanual import primitives as P


def run_spec(sim: P.Sim, spec: str):
    """Run one skill:arm[:object[:x:y]] spec on an already-reset sim; returns (success, metrics)."""
    parts = spec.split(":")
    skill, arm = parts[0], sim.arms[parts[1].upper()]
    if skill == "pick":
        return P.pick(sim, arm, parts[2])
    if skill == "place":   # place:arm:object:x:y[:free]  ("free" = hand-off, no yaw alignment)
        return P.place(sim, arm, parts[2], (float(parts[3]), float(parts[4])), align=(len(parts) < 6 or parts[5] != "free"))
    if skill == "open_drawer":
        return P.open_drawer(sim, arm)
    if skill == "home":
        P.go_home(sim, arm)
        return True, {}
    raise SystemExit(f"unknown skill {skill}")


def run_sequence(sim: P.Sim, specs: list[str], seed: int):
    """Reset once, then run each skill; returns [(spec, success, metrics)]."""
    first_obj = next((s.split(":")[2] for s in specs if len(s.split(":")) > 2 and s.startswith("pick")), None)
    sim.reset(seed, jitter={first_obj: 0.02} if first_obj else None)
    results = []
    for spec in specs:
        ok, metrics = run_spec(sim, spec)
        results.append((spec, ok, metrics))
        print(f"{spec}: success={ok} metrics={ {k: round(v, 3) for k, v in metrics.items()} }")
    return results


def main(argv):
    seed = 101
    if argv and argv[0] == "--seed":
        seed, argv = int(argv[1]), argv[2:]
    specs = argv
    sim = P.Sim()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.55, 180, -30
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    frames = []
    focus = {"body": sim.body("sliding_tray")}

    def hook(s):
        cam.lookat[:] = s.data.xpos[focus["body"]]
        renderer.update_scene(s.data, camera=cam)
        frames.append(renderer.render().copy())
    sim.frame_hook = hook
    # follow the object of the current skill
    orig = {"pick": P.pick, "place": P.place, "open_drawer": P.open_drawer}

    def focused(fn, name):
        def wrapper(sim_, arm, *a, **k):
            focus["body"] = sim_.body(a[0]) if name != "open_drawer" else sim_.body("sliding_tray")
            return fn(sim_, arm, *a, **k)
        return wrapper
    for name, fn in orig.items():
        setattr(P, name, focused(fn, name))
    try:
        results = run_sequence(sim, specs, seed)
    finally:
        for name, fn in orig.items():
            setattr(P, name, fn)

    out = Path("outputs/review")
    out.mkdir(parents=True, exist_ok=True)
    stem = "__".join(s.replace(":", "_") for s in specs)[:80]
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    tiles = []
    for i in idx:
        im = frames[i].copy()
        cv2.putText(im, f"frame {i}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 40, 40), 2)
        tiles.append(im)
    Image.fromarray(np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])).save(out / f"{stem}_closeups.png")
    with av.open(str(out / f"{stem}_slow.mp4"), "w") as c:
        s = c.add_stream("h264", rate=8)
        s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    print(f"{len(frames)} frames -> {out / (stem + '_closeups.png')} and {out / (stem + '_slow.mp4')}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
