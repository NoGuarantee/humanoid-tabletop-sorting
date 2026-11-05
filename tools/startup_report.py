#!/usr/bin/env python3
"""Start-up timeline from a layout log of `remote.sh eval --layouts N` (t=0 is the line `[bench] layout start`).

Wall time comes from the ROS log prefix. Lines with `sim X` give the sim clock; other moments are interpolated between
them, and everything before the first such line is extrapolated at the mean rate, so the very start of sim time is an
estimate. Usage: tools/startup_probe.py.
"""
import re
import sys

LINE = re.compile(r"\[(INFO|WARN|ERROR)\] \[(\d+\.\d+)\] \[([\w.]+)\]: (.*)")
EVENTS = [
    (r"motion server ready", "motion_node up"),
    (r"startup: preload starts", "preload starts"),
    (r"startup: first /joint_states", "first /joint_states"),
    (r"curobo ready in", "MotionPlanner built"),
    (r"startup: planner loaded", "preload finished"),
    (r"\d+ waypoints", "plan ready"),
    (r"move_tcp \w+ \w+: tcp", "move_tcp arrived"),
    (r"published /sim/task/done", "done published"),
]


def report(path):
    t0 = None
    clock = []  # (wall, sim)
    events = []
    with open(path, errors="replace") as f:
        for line in f:
            m = re.search(r"\[bench\] layout start (\d+\.\d+)", line)
            if m:
                t0 = float(m.group(1))
                continue
            m = LINE.search(line)
            if not m:
                continue
            wall, text = float(m.group(2)), m.group(4)
            s = re.search(r"sim (\d+\.\d+)$", text)
            if s and float(s.group(1)) > 0.0:
                clock.append((wall, float(s.group(1))))
            if text.startswith("probe: clock"):
                continue
            for pattern, label in EVENTS:
                if re.search(pattern, text):
                    events.append((wall, label, text))
    if t0 is None:
        sys.exit(f"{path}: no `[bench] layout start` line")
    clock.sort()
    rate = (clock[-1][1] - clock[0][1]) / (clock[-1][0] - clock[0][0]) if len(clock) > 1 else 1.0

    def sim(wall):
        if not clock or wall <= clock[0][0]:
            return (clock[0][1] if clock else 0.0) - rate * ((clock[0][0] if clock else wall) - wall)
        for (w0, s0), (w1, s1) in zip(clock, clock[1:]):
            if wall <= w1:
                return s0 + (s1 - s0) * (wall - w0) / max(w1 - w0, 1e-6)
        return clock[-1][1] + rate * (wall - clock[-1][0])

    print(f"{path}: mean sim/wall rate {rate:.2f}, {len(clock)} clock samples")
    print(f"{'wall s':>8} {'sim s':>8}  event")
    for wall, label, text in events:
        print(f"{wall - t0:8.1f} {sim(wall) - sim(t0):8.1f}  {label}: {text[:90]}")


for name in sys.argv[1:]:
    report(name)
