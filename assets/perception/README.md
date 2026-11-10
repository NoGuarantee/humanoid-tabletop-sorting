# RGB-D model bundle

`geometry_only_ros_bundle.tar.gz` is the existing Team 049 RGB-D release asset
from `pr2-rgbd-model-20261105`, copied without modification so a clean evaluator
checkout can stage it without GitHub credentials or runtime network access.

- Source: https://github.com/nssr-hackathon/nssr-sol-team-049/releases/tag/pr2-rgbd-model-20261105
- Bytes: 12,224,110
- SHA-256: `b33b2debe3c1710bd6e7458b983767bfc7a7ce003b0d9c5560ad20ec5bd7fade`

`setup.sh` uses `tools/stage_perception_rgbd.sh` to verify the archive and stage
`weights/rgbd-b33b2debe3c1`. An explicitly supplied archive under `weights/`
continues to take precedence and must have the same pinned checksum.
If neither local archive exists, `NSSR_RGBD_BUNDLE_URL` can provide the same
pinned archive (bounded curl); the default private release is not anonymously
accessible, so the checked-in archive is the normal evaluator path.

The bundle supplies sensor-model recognition and canonical RGB-D frames; it is
not a complete payload envelope or a measured table/tray layout. Grasp still
requires separate sensor measurements with their own provenance.
