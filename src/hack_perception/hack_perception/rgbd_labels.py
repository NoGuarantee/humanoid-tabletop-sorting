"""Class order of the frozen segmentation weights and ROS category names."""

CLASS_NAMES = ("figure", "dish", "box", "electronics", "falcon50", "falcon15", "tray", "rack")
CLASS_BY_ID = dict(enumerate(CLASS_NAMES, 1))
TRAY_CLASS = CLASS_NAMES.index("tray") + 1

POOL_IDENTITIES = {
    4: {1: (0, 1, 2), 2: (3, 4, 5), 3: (6, 7, 8), 5: (15,), 6: (16,)},
    6: {3: (12, 13, 14), 4: (9, 10, 11)},
}


def validate_detector_labels(labels):
    if labels != dict(enumerate(CLASS_NAMES)):
        raise ValueError(f"unexpected YOLO class order: {labels}")
