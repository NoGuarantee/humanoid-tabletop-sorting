import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
rclpy.init(args=[]);node=Node('own_camera_readiness_gate');seen=[]
node.create_subscription(Image,'/cam_head/color/image_raw',lambda message:seen.append(message.header.stamp.sec),2)
deadline=time.monotonic()+360
while not seen and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.5)
node.destroy_node();rclpy.shutdown()
if not seen:raise SystemExit(2)
print('own_camera_ready',seen[-1],flush=True)
