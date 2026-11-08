#!/usr/bin/env python3
"""Isolated ROS middleware test of production Clients/FSM using synthetic sensors.

No simulator or physical robot. Refuses the competition domain. Run after colcon:
ROS_DOMAIN_ID=196 ROS_LOCALHOST_ONLY=1 python3 tools/fsm_ros_smoke.py
"""
import os
if os.environ.get('ROS_DOMAIN_ID') in (None,'146'):
    raise SystemExit('Set an isolated ROS_DOMAIN_ID (never 146)')
import threading
import time
from queue import Queue, Empty

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rosgraph_msgs.msg import Clock
from hack_interfaces.action import Hand, MoveTcp, MoveJoints
from hack_interfaces.srv import Perceive
from hack_interfaces.msg import DetectedObject, Container
from hack_motion.robot import JOINTS, HAND_PRESETS
from hack_brain.brain_node import Brain
from hack_brain.context import Context, Event, State, Config
from hack_brain.fsm import transition
from hack_brain.task import parse_task


class Rig(Node):
    def __init__(self):
        super().__init__('fsm_smoke_rig', use_global_arguments=False)
        self.sim=10.;self.closed=False;self.placed=False;self.z=.2;self.failed=False
        self.calls=[];self.cb=ReentrantCallbackGroup()
        self.clock=self.create_publisher(Clock,'/clock',10)
        self.joints=self.create_publisher(JointState,'/joint_states',10)
        self.hands={s:self.create_publisher(JointState,f'/inspire_hand/state/{s}_hand',10) for s in ('left','right')}
        self.tf=TransformBroadcaster(self)
        self.create_timer(.02,self.sample)
        self.create_service(Perceive,'perception/perceive',self.scene,callback_group=self.cb)
        self.servers=[ActionServer(self,a,'motion/'+name,fn,callback_group=self.cb,
                                  cancel_callback=lambda _:CancelResponse.ACCEPT)
                      for a,name,fn in [(Hand,'hand',self.hand),(MoveTcp,'move_tcp',self.tcp),
                                        (MoveJoints,'move_joints',self.move_joints)]]

    def stamp(self):
        from builtin_interfaces.msg import Time
        return Time(sec=int(self.sim),nanosec=int((self.sim-int(self.sim))*1e9))

    def sample(self):
        self.sim+=.05
        self.clock.publish(Clock(clock=self.stamp()))
        msg=JointState();msg.header.stamp=self.stamp();msg.name=list(JOINTS);msg.position=[0.]*len(JOINTS)
        self.joints.publish(msg)
        for side,pub in self.hands.items():
            msg=JointState();msg.header.stamp=self.stamp();msg.name=[str(i) for i in range(1,7)]
            msg.position=[.4,.4,.4,.4,.7,.2] if self.closed and side=='left' else [1.]*6
            msg.effort=[.3]*6 if self.closed and side=='left' else [0.]*6
            pub.publish(msg)
            tf=TransformStamped();tf.header.stamp=self.stamp();tf.header.frame_id='pelvis'
            tf.child_frame_id=side+'_tcp_link';tf.transform.translation.z=self.z
            tf.transform.rotation.w=1.;self.tf.sendTransform(tf)

    def scene(self,req,res):
        res.success=True;res.scene.header.frame_id='pelvis';res.scene.header.stamp=self.stamp();res.scene.table_z=0.
        obj=DetectedObject();obj.name='a';obj.track_id='track';obj.category='figure';obj.confidence=.95
        obj.center.x=.4 if not self.placed else .6;obj.center.y=.2;obj.center.z=.03
        obj.size.x=obj.size.y=.03;obj.size.z=.06;obj.top_z=.06
        tray=Container();tray.id='tray';tray.kind='tray';tray.pose.position.x=.6;tray.pose.position.y=.2
        tray.pose.position.z=.07;tray.pose.orientation.w=1.;tray.size.x=.22;tray.size.y=.32;tray.size.z=.07
        res.scene.objects=[obj];res.scene.containers=[tray]
        return res

    def move_joints(self,h):
        self.calls.append('look');h.succeed();return MoveJoints.Result(success=True)

    def hand(self,h):
        self.calls.append(h.request.preset)
        if h.request.preset=='open':
            if self.closed:self.placed=True
            self.closed=False
        else:self.closed=True
        h.succeed();return Hand.Result(success=True)

    def tcp(self,h):
        self.calls.append(h.request.mode)
        if not self.failed:
            self.failed=True;h.succeed();return MoveTcp.Result(success=False,message='synthetic unreachable')
        self.z=h.request.target.position.z
        h.succeed();return MoveTcp.Result(success=True)


def main():
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true']);rig=Rig()
    node=Brain()
    executor=MultiThreadedExecutor(num_threads=6);executor.add_node(rig);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    mission=None
    try:
        until=time.monotonic()+10
        while not all(x.server_is_ready() for x in node.adapters.actions.values()) or node.get_clock().now().nanoseconds==0:
            if time.monotonic()>until:raise AssertionError('discovery timeout')
            time.sleep(.05)
        node.emit(Event('TASK',data='{"task":"sort2","objects":{"a":"figure","b":"figure"},"categories":{"figure":"tray"}}'))
        mission=threading.Thread(target=node.mission,daemon=True);mission.start();mission.join(timeout=35)
        c=node.mission_context
        assert not mission.is_alive(),(c.state,c.reason)
        assert c.state==State.DONE,(c.state,c.reason)
        assert c.history.get('a') and c.history['a'].placed,(c.reason,rig.calls)
        assert c.recoveries>=1 and rig.calls.count('open')>=3,rig.calls
        assert node.done_attempts==3,node.done_attempts
        print('PASS: real Brain constructor/mission, ROS adapters, measured recovery, lift, release, visual placement')
    finally:
        if mission and mission.is_alive():
            node.emit(Event('CANCEL',data='smoke_timeout'));mission.join(timeout=8)
        executor.shutdown(timeout_sec=3);thread.join(timeout=3)
        node.destroy_node();rig.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
