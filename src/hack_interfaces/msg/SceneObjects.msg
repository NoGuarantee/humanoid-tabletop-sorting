# Result of one perception pass.
std_msgs/Header header     # stamp of the camera frame, frame_id = pelvis
float32 table_z            # height of the table plane, 0 if not found
DetectedObject[] objects
Container[] containers
