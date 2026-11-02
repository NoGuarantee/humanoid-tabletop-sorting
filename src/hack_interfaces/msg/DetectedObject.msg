# One object found by perception. All coordinates are in the `pelvis` frame, metres and radians.
string track_id            # stable between Perceive calls while the object has not moved
string name                # exact object name from /sim/objects/names, "" if not sure
string category            # jar / box / figure / dish / electronics / falcon50 / falcon15
string cap_color           # tubes only: red / blue / orange ...
float32 confidence         # calibrated classification confidence, 0..1
geometry_msgs/Point center # centre of the visible part
float32 yaw                # main axis in the table plane (PCA), [-pi/2, pi/2]
geometry_msgs/Vector3 size # extent along the main axis / across it / height
float32 top_z              # highest point of the object
bool lying                 # tubes only: the tube lies on the table
bool partially_visible     # cut by the image border or occluded
