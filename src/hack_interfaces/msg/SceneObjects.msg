# Result of one perception pass.
std_msgs/Header header     # stamp of the camera frame, frame_id = pelvis
float32 table_z            # height of the table plane, 0 if not found
float64[] table_plane      # a, b, c, d: a x + b y + c z + d = 0 in pelvis, (a, b, c) unit and up; empty if not found
DetectedObject[] objects
Container[] containers
