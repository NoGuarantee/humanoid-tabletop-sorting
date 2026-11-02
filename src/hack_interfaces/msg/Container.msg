# A sorting tray or a tube rack found by perception, `pelvis` frame.
string id                  # tray_left / tray_right / tray_center / table_r50_0 / hood_r15_1 ...
string kind                # tray | rack
string color               # red / blue / green / white ...
geometry_msgs/Pose pose    # centre of the opening; yaw of the container in the orientation
geometry_msgs/Vector3 size
geometry_msgs/Point[] slots   # racks only: centres of the slot openings
bool[] slot_occupied          # racks only, same order as slots
