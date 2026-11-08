"""Real ROS CPU replay; saved sensors repeatedly stamped on wall clock, no simulator."""
import json
from pathlib import Path
import threading
from time import monotonic, sleep
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster
from std_srvs.srv import Trigger
from hack_interfaces.srv import Perceive
from hack_perception.perception_node import PerceptionNode


def main():
    rclpy.init(args=['--ros-args', '-p', 'rgbd_model_dir:=/models', '-p', 'rgbd_enable_model_poses:=true',
        '-p', 'rgbd_pool:=4', '-p', 'rgbd_on_demand:=true', '-p', 'rgbd_model_names:=Schleich_African_Black_Rhino,Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g'])
    perception=PerceptionNode()
    source=Node('sensor_replay_source')
    client=Node('sensor_replay_client')
    sensors=[]
    static=StaticTransformBroadcaster(source)
    transforms=[]
    for camera in ('cam_head','cam_torso'):
        data=np.load('/sensors/center_'+camera+'.npz')
        pubs={kind:source.create_publisher(typ, '/'+camera+'/'+topic, 2) for kind,typ,topic in (
            ('rgb',Image,'color/image_raw'),('depth',Image,'depth/image_raw'),
            ('color_info',CameraInfo,'color/camera_info'),('depth_info',CameraInfo,'depth/camera_info'))}
        sensors.append((camera,data,pubs))
        tf=TransformStamped();tf.header.frame_id='pelvis';tf.child_frame_id=camera+'_optical'
        matrix=data['T_pelvis_camera'];tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z=map(float,matrix[:3,3])
        q=Rotation.from_matrix(matrix[:3,:3]).as_quat()
        tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z,tf.transform.rotation.w=map(float,q)
        transforms.append(tf)
    static.sendTransform(transforms)
    def publish():
        stamp=source.get_clock().now().to_msg()
        for camera,data,pubs in sensors:
            h,w=data['depth'].shape
            for kind in ('rgb','depth'):
                msg=Image();msg.header.stamp=stamp;msg.header.frame_id=camera+'_optical'
                msg.height=h;msg.width=w;msg.encoding='rgb8' if kind=='rgb' else '16UC1'
                msg.step=w*(3 if kind=='rgb' else 2);msg.data=data[kind].astype(np.uint8 if kind=='rgb' else '<u2').tobytes()
                pubs[kind].publish(msg)
            for kind in ('color_info','depth_info'):
                msg=CameraInfo();msg.header.stamp=stamp;msg.header.frame_id=camera+'_optical'
                msg.height=h;msg.width=w;msg.k=list(map(float,data['K'].ravel()));pubs[kind].publish(msg)
    timer=source.create_timer(.1,publish)
    executor=MultiThreadedExecutor(num_threads=4)
    for node in (source,perception,client):executor.add_node(node)
    thread=threading.Thread(target=executor.spin);thread.start()
    pose=client.create_client(Trigger,'/perception/estimate_model_poses');scene=client.create_client(Perceive,'/perception/perceive')
    def call(service,request):
        start=monotonic();future=service.call_async(request)
        while not future.done() and monotonic()-start<10:sleep(.005)
        if not future.done():raise RuntimeError('RPC timeout')
        return future.result(),monotonic()-start
    results=[]
    try:
        deadline=monotonic()+30
        while perception.model_poses.state=='worker_loading' and monotonic()<deadline:sleep(.1)
        call(pose,Trigger.Request())
        for _ in range(20):
            busy=perception.model_poses.pending
            result,elapsed=call(scene,Perceive.Request(camera='head',region='table'))
            stamp=result.scene.header.stamp.sec*10**9+result.scene.header.stamp.nanosec
            results.append(dict(phase='normal',pending=busy[0] if busy else None,wall_s=elapsed,
                success=result.success,message=result.message,trays=len(result.scene.containers),
                age_s=(client.get_clock().now().nanoseconds-stamp)*1e-9))
            result,_=call(pose,Trigger.Request())
            payload=json.loads(result.message)
            if result.success:break
        worker=perception.model_poses.worker
        worker.process.terminate();worker.process.join(timeout=2)
        # A separate real process death exercises fallback and restart.
        result,elapsed=call(scene,Perceive.Request(camera='head',region='table'))
        results.append(dict(phase='worker_killed',wall_s=elapsed,success=result.success,message=result.message))
        receipt=dict(scope='ROS/DDS CPU replay; repeated saved sensor frames stamped on wall clock; no live LOOK_TABLE, motion or solution.sh',
            scenes=results,last_pose=payload,worker_restarts=perception.model_poses.worker_restarts)
        Path('/trial/ros-replay.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    finally:
        timer.cancel();executor.shutdown();thread.join(timeout=3)
        perception.destroy_node();source.destroy_node();client.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
