"""LOOK_TABLE then the very first Perceive RPC; only ROS sensors and metadata."""
import json,os,time
from pathlib import Path
import threading
from time import monotonic,sleep
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from rcl_interfaces.msg import Log
from hack_interfaces.action import MoveJoints
from hack_interfaces.srv import Perceive
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy


def main():
 rclpy.init(args=[]);node=Node('perceive_live_probe',parameter_overrides=[Parameter('use_sim_time',value=True)])
 q={};task={};names={};pose_started=threading.Event()
 def rosout(message):
  if message.name=="hack_perception" and '"event": "rgbd_pose_submitted"' in message.msg:
   pose_started.set()
 node.create_subscription(Log,"/rosout",rosout,100)
 node.create_subscription(JointState,'/joint_states',lambda m:q.update(zip(m.name,m.position)),10)
 latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
 def metadata(dst,msg):dst.clear();dst.update(json.loads(msg.data))
 node.create_subscription(String,'/sim/task',lambda m:metadata(task,m),latched)
 node.create_subscription(String,'/sim/objects/names',lambda m:metadata(names,m),latched)
 executor=MultiThreadedExecutor(num_threads=2);executor.add_node(node)
 thread=threading.Thread(target=executor.spin);thread.start()
 motion=ActionClient(node,MoveJoints,'/motion/move_joints')
 perceive=node.create_client(Perceive,'/perception/perceive')
 pose=node.create_client(Trigger,'/perception/estimate_model_poses')
 def wait(future,timeout=30):
  start=monotonic()
  while not future.done() and monotonic()-start<timeout:sleep(.002)
  if not future.done():raise RuntimeError('future timeout')
  return future.result()
 def move(pitch):
  goal=MoveJoints.Goal(joints=['head_pitch_joint','head_yaw_joint','body_yaw_joint'],positions=[pitch,0.,0.],speed=.6)
  handle=wait(motion.send_goal_async(goal));assert handle.accepted
  result=wait(handle.get_result_async()).result
  if not result.success:raise RuntimeError('LOOK action failed: '+result.message)
  return dict(stamp_ns=node.get_clock().now().nanoseconds,joints={k:q.get(k) for k in goal.joints},max_error=result.max_error)
 rows=[]
 try:
  assert motion.wait_for_server(timeout_sec=90)
  assert perceive.wait_for_service(timeout_sec=30)
  deadline=monotonic()+240
  while not q or not names or not task:
   if monotonic()>deadline:raise RuntimeError('sensor metadata timeout')
   sleep(.1)
  assert task.get('objects_from')=='pool 4'
  graph_deadline=monotonic()+25
  while node.get_node_names().count('hack_perception')!=1 and monotonic()<graph_deadline:sleep(.1)
  assert node.get_node_names().count('hack_perception')==1, repr(node.get_node_names())
  Path('/evidence/task_metadata.json').write_text(json.dumps(dict(task=task,names=names),indent=2)+'\n')
  for repeat in range(int(os.environ.get("NSSR_PROBE_REPEATS","6"))):
   move(.05 if repeat%2 else .15)
   look=move(.4)
   if repeat>=3 and os.environ.get("NSSR_DURING_6D","0")=="1" and pose.wait_for_service(timeout_sec=.1):
    pose_started.clear()
    trigger=wait(pose.call_async(Trigger.Request()))
    if trigger.success:
     trigger=wait(pose.call_async(Trigger.Request()))
    print("pose_trigger",trigger.message,flush=True)
    if not pose_started.wait(timeout=4):raise RuntimeError('no real pose submission observed')
   request_stamp=node.get_clock().now().nanoseconds
   wall_started=time.time();started=monotonic();result=wait(perceive.call_async(Perceive.Request(camera='head',region='table')));elapsed=monotonic()-started
   scene=result.scene;stamp=scene.header.stamp.sec*10**9+scene.header.stamp.nanosec
   trays={t.id for t in scene.containers}
   cats=task.get('categories',{})
   objects=[dict(name=o.name,category=o.category,confidence=o.confidence,partial=o.partially_visible) for o in scene.objects]
   compatible=[o for o in objects if o['category'] in cats and cats[o['category']] in trays and o['confidence']>=.6 and not o['partial']]
   row=dict(repeat=repeat,look=look,request_stamp_ns=request_stamp,stamp_ns=stamp,
    wall_s=elapsed,wall_started=wall_started,wall_finished=time.time(),age_sim_s=(node.get_clock().now().nanoseconds-stamp)*1e-9 if stamp else None,
    during_6d=repeat>=3 and os.environ.get("NSSR_DURING_6D","0")=="1",success=result.success,message=result.message,trays=sorted(trays),objects=objects,compatible_candidates=compatible)
   row['passed']=bool(result.success and stamp>=request_stamp and row['age_sim_s']<=1. and trays and objects and compatible)
   rows.append(row);Path('/evidence/live_result.json').write_text(json.dumps(dict(rows=rows,passed=all(r['passed'] for r in rows)),indent=2)+'\n')
   print(json.dumps(row),flush=True)
 finally:
  executor.shutdown();thread.join(timeout=3);node.destroy_node();rclpy.shutdown()
 if not all(r['passed'] for r in rows):raise SystemExit(2)

if __name__=='__main__':main()
