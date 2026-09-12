# stage2_perception — owner: Lauren
Camera frame -> object + drawer positions in tabletop coordinates (stubbed; OpenCV later).
Input: camera frame (or None in sim). Output: `common.types.SceneState`.
Run standalone from repo root:
`python3 -c "from stage2_perception import perceive; print(perceive())"`
