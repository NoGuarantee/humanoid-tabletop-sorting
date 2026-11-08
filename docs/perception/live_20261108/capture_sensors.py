"""Save one live registered RGB-D exposure and its measured camera TF only."""
from pathlib import Path
from time import monotonic
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image,CameraInfo
from tf2_ros import Buffer,TransformListener
from hack_perception.rgbd_frames import CAMERAS,STREAMS,PairBuffer,decode_pair
from hack_perception.geometry import transform_matrix
from hack_perception.cdr_image import ros_image
from PIL import Image as PILImage
rclpy.init(args=[]);node=Node('own_sensor_evidence_capture',parameter_overrides=[Parameter('use_sim_time',value=True)]);tf=Buffer();listener=TransformListener(tf,node)
buffer=PairBuffer()
for camera in CAMERAS:
 for kind,stream in STREAMS.items():
  node.create_subscription(CameraInfo if kind.endswith('info') else Image,'/'+camera+'/'+stream,
    lambda message,camera=camera,kind=kind:buffer.add(camera,kind,ros_image(message) if not kind.endswith("info") else message),2,raw=not kind.endswith("info"))
deadline=monotonic()+15
while monotonic()<deadline:
 rclpy.spin_once(node,timeout_sec=.05)
 pair=buffer.latest(node.get_clock().now().nanoseconds,2_000_000_000)
 if pair is None:continue
 stamp,views=pair
 try:
  sensors=decode_pair(views,lambda header:transform_matrix(tf.lookup_transform('pelvis',header.frame_id,Time.from_msg(header.stamp))))
 except Exception:continue
 for camera,sensor in zip(CAMERAS,sensors):
  np.savez_compressed('/evidence/'+camera+'.npz',rgb=sensor['rgb'],depth=sensor['depth'],K=sensor['K'],T_pelvis_camera=sensor['T'],stamp_ns=stamp)
  PILImage.fromarray(sensor['rgb']).save('/evidence/'+camera+'.png')
 print('saved_sensor_stamp_ns',stamp,flush=True);break
else:raise SystemExit('no complete measured exposure')
node.destroy_node();rclpy.shutdown()
