"""Table and tray poses measured in the simulator, for the probes in tools/ only. Never import this from src/.

The solution takes the table plane and the trays from perception (perception/scene, SceneObjects.table_plane and
containers). These are the numbers the probes ran with before that, so they can still build a world without a scan.

The table in pelvis: z = -0.1792 + 0.0700 x, 4 degrees. Fitted to depth (rms 0.3 mm), and the open fingers touch it
there at x 0.28, 0.38 and 0.50 (tools/table_touch.py). The cameras give the same plane: -0.1793 + 0.0701 x on the head
frame and -0.1802 + 0.0710 x on the torso frame of docs/perception/live_20261108.
The trays (hackathon/config/scene.yaml): x, y from the pelvis along the table, 1 mm over the top. Depth frames of
layout 1 agree within 3 mm.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "hack_motion"))

from hack_motion.robot import TRAY_SIZE, world_scene  # noqa: E402

_A, _B = -0.1792, 0.0700  # z = a + b x
_N = math.sqrt(1.0 + _B * _B)
TABLE = (-_B / _N, 0.0, 1.0 / _N, -_A / _N)  # a x + b y + c z + d = 0, as SceneObjects.table_plane
_TILT = math.atan(_B)
_ABOVE_PELVIS = -_A * math.cos(_TILT)  # m, pelvis origin over the table top along the table normal

TRAYS = {"tray_left": (0.34, 0.69), "tray_right": (0.34, -0.69), "tray_center": (0.53, 0.0)}  # along the table, y


def table_z(x):
    """z of the table top in pelvis at this x."""
    return _A + _B * x


def on_table(along, y, height):
    """Pelvis (x, y, z) of a point `along` m out on the table axis from the pelvis, `height` m above the top."""
    c, s = math.cos(_TILT), math.sin(_TILT)
    z_level = height - _ABOVE_PELVIS
    return along * c - z_level * s, y, along * s + z_level * c


def tray_rim(name):
    """Pelvis (x, y, z) of the middle of the opening of a tray."""
    return on_table(*TRAYS[name], TRAY_SIZE[2])


def tray(name):
    """The tray as perception reports it: (id, x, y, z of the opening, yaw, size x, y, z)."""
    return (name, *tray_rim(name), 0.0, *TRAY_SIZE)


def trays(names=None):
    return tuple(tray(name) for name in sorted(TRAYS if names is None else names))


def table_scene(names=()):
    """The cuRobo world of the table and these trays, the way the planner builds it from a scan."""
    return world_scene(TABLE, trays(names))[0]


def use(planner, names=None):
    """Give a CuroboPlanner the measured table and trays (all three by default) instead of a scan."""
    planner.table = TABLE
    planner.trays = trays(names)
