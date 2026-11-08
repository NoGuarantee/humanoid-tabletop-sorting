#!/usr/bin/env python3
"""Real motion action cancellation and measured-stop adapter, isolated synthetic joints."""
import os
if os.environ.get('ROS_DOMAIN_ID') in (None,'146'):
    raise SystemExit('Refusing competition ROS domain')
import threading
import time
from queue import Queue, Empty
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from bodyctrl_msgs.msg import CmdSetMotorPosition
from hack_motion.motion_node import MotionNode
from hack_motion.robot import JOINTS
from hack_brain.clients import Clients
from hack_brain.context import Context, State, Effect, Operation


class Rig(Node):
    def __init__(self):
        super().__init__('cancel_smoke_rig', use_global_arguments=False)
        self.sim=10.;self.commands=[]
        self.clock=self.create_publisher(Clock,'/clock',10)
        self.joints=self.create_publisher(JointState,'/joint_states',10)
        self.create_subscription(CmdSetMotorPosition,'/head/cmd_pos',self.command,10)
        self.create_timer(.02,self.sample)
    def command(self,m):self.commands.extend((c.name,c.pos) for c in m.cmds)
    def sample(self):
        from builtin_interfaces.msg import Time
        self.sim+=.05;t=Time(sec=int(self.sim),nanosec=int((self.sim-int(self.sim))*1e9))
        self.clock.publish(Clock(clock=t))
        msg=JointState();msg.header.stamp=t;msg.name=list(JOINTS);msg.position=[0.]*len(JOINTS)
        self.joints.publish(msg)


def main():
    rclpy.init(args=['--ros-args','-p','planner:=ik','-p','use_sim_time:=true'])
    rig=Rig();motion=MotionNode();node=Node('cancel_smoke_client')
    q=Queue();clients=Clients(node,q.put,{})
    executor=MultiThreadedExecutor(num_threads=6)
    for n in (rig,motion,node):executor.add_node(n)
    worker=threading.Thread(target=executor.spin,daemon=True);worker.start()
    try:
        deadline=time.monotonic()+10
        while not clients.actions['joints'].server_is_ready() or clients.now()==0:
            if time.monotonic()>deadline:raise AssertionError('discovery failed')
            time.sleep(.05)
        c=Context(state=State.LOOK_TABLE,sim=clients.now(),active=Operation(1,State.LOOK_TABLE,clients.now(),time.monotonic()))
        clients.execute(Effect('RUN',1,State.LOOK_TABLE),c)
        events=[];cancelled=False;deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            clients.transport.poll()
            try:
                event=q.get(timeout=.05);events.append(event)
            except Empty:continue
            if event.kind=='ACCEPTED' and not cancelled:
                # Let the server issue its initial position target before cancelling it.
                end=time.monotonic()+2
                while not rig.commands and time.monotonic()<end:time.sleep(.02)
                clients.execute(Effect('CANCEL',1),c);cancelled=True
            if event.kind=='RESULT':break
        assert any(e.kind=='CANCEL_ACK' and e.ok for e in events),events
        result=next(e for e in events if e.kind=='RESULT')
        assert not result.ok and result.stopped,result
        assert clients.transport.pending is None
        assert any(pos>.1 for _,pos in rig.commands),rig.commands
        assert rig.commands[-1][1]==0,rig.commands
        print('PASS: real MotionNode cancels, replaces targets with measured hold, adapter confirms fresh rest')
    finally:
        executor.shutdown(timeout_sec=3);worker.join(timeout=3)
        for n in (node,motion,rig):n.destroy_node()
        rclpy.try_shutdown()

if __name__=='__main__':main()
