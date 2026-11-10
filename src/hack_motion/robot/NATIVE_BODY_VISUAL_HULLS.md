# Native body visual hulls

`native_body_visual_hulls.json` is a robot-only perception model, not an object
model or a replacement for motion collision spheres. The self-filter loader
pins its SHA-256, the existing hand model and their common native robot source.

- Source robot USD SHA-256: `0e99d7d34ed7b4bd6aef0a0246c5d6f0e8d68b99a567a3e836f13cbdef1331e5`.
- Asset SHA-256: `9cbb5f1c8340b89fc179cf765619938cafb2d2d0dc6de2bbdb574b0fe1ab4ebe`.
- Fourteen visual hulls: the eight original elbow/wrist hulls plus six shoulder
  pitch/roll/yaw hulls added after the E68 shoulder-fringe failure. Hidden
  collision proxies were excluded. This is the unchanged asset from PR9
  `e499c89d092bf2c5f05a9975c05034cef11b3373`.
- Fifteen native body joint frames reconstruct the authored zero-angle frames
  within 19.7 nm. Native body frames must be used; URDF wrist origins differ
  by approximately 2.982 mm in the measured qualification snapshot.

Runtime uses measured joint angles and exposure camera calibration. It loads
no USD, simulator state, object CAD or saved scene layout. The Euclidean 3 mm
margin uses exact point-to-triangle distance at edges and corners. Exclusion
remains unknown depth, not free space or positive object identity. A contacting
object can lose pixels inside the conservative robot envelope.

The JSON includes mesh paths, transforms, source counts, triangle-index hashes,
convex-containment proofs and the full native joint description. The generator
and independent PXR export, convex-QP distance check and raw-pixel replay are
preserved in the project qualification artifacts under
`outputs/grasp_pr9_live_20261108/robot_visual_hull_audit_20261109`.
The shoulder extension and replay are under `episode69/robot_hulls/` and
`episode69/refilter_report.json` in the same archive. Tests retain the original
135-point fixture and the E68 shoulder/target surfaces.
