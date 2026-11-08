"""ROS adapter. Callbacks only enqueue events; no callback waits on a future."""
import math

from hack_interfaces.action import Hand, MoveJoints, MoveTcp
from hack_interfaces.srv import Perceive
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from .context import Container, Event, Evidence, Object, Scene, State
from .physical import PhysicalVerifier
from .sensors import Measurements, stamp
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from hack_motion.robot import HAND_PRESETS, JOINTS
from .fsm import VERIFY
from .skills import Skills
from .transport import ActionTransport


class Clients:
    # Sorting evidence is produced by PhysicalVerifier, never by action success.
    # Tube skills and rack verification remain unavailable.
    capabilities = frozenset({'perception', 'pick', 'transfer', 'release', 'retreat', 'place', 'verification'})
    unavailable = 'sorting sensor verification enabled; tube skills unavailable'

    def __init__(self, node, emit, db):
        self.node, self.emit = node, emit
        cb = ReentrantCallbackGroup()
        self.perceive_cli = node.create_client(Perceive, 'perception/perceive', callback_group=cb)
        self.actions = {
            'tcp': ActionClient(node, MoveTcp, 'motion/move_tcp', callback_group=cb),
            'joints': ActionClient(node, MoveJoints, 'motion/move_joints', callback_group=cb),
            'hand': ActionClient(node, Hand, 'motion/hand', callback_group=cb),
        }
        self.measurements = Measurements()
        for key, topic in [('joints','/joint_states'), ('left','/inspire_hand/state/left_hand'),
                           ('right','/inspire_hand/state/right_hand')]:
            node.create_subscription(JointState, topic, lambda msg,k=key: self.measurements.update(k,msg),
                                     10, callback_group=cb)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf,node)
        self.verifier = PhysicalVerifier(self.measurements,self.tcp)
        self.stop_resources = {}
        self.scene_pending = None
        self.scene_future = None
        self.next_scene_request = 0.0
        self.motion_wait = None
        self.verify_pending = None
        self.verify_scene = None
        self.verify_future = None
        self.transport = ActionTransport(emit,self.stopped,self.now)
        self.skills = Skills(db)
        self.scenes = {}

    def execute(self, effect, context):
        if effect.kind == 'CANCEL':
            if self.motion_wait is not None and self.motion_wait.op_id == effect.op_id:
                self.motion_wait = None
                self.emit(Event('CANCEL_ACK',effect.op_id))
                self.emit(Event('RESULT',effect.op_id,ok=False,stopped=True))
                return
            if context.active.phase in VERIFY or context.active.phase in (State.OBSERVE, State.DESTINATION, State.REFRESH_TARGET):
                # Read-only operations can be abandoned. Late responses retain old IDs.
                self.verify_pending = None
                self.emit(Event('CANCEL_ACK', effect.op_id))
                self.emit(Event('RESULT', effect.op_id, ok=False, stopped=True))
            else:
                self.transport.cancel(effect.op_id)
            return
        if effect.phase in VERIFY:
            self.verify_pending = effect
            self.verify_scene = None
            self.verify_future = None
            return
        if effect.phase in (State.OBSERVE, State.DESTINATION, State.REFRESH_TARGET):
            self.refresh_target = effect.target.id if effect.target else ''
            self.scene_pending = effect
            self.scene_future = None
            self.next_scene_request = 0.0
            return
        try:
            # Only owner thread updates this cache from result payloads via accept_scene.
            scene = self.scenes.get('destination' if effect.phase == State.TRANSFER else 'table')
            name, goal = self.skills.command(effect, scene, context.history[effect.target.id].attempts if effect.target else 0)
            if not self.actions[name].server_is_ready():
                self.motion_wait = effect
            else:
                self.motion_wait = None
                if effect.phase == State.OPEN:
                    self.verifier.arm = goal.hand
                    self.verifier.target_id = effect.target.id
                    self.verifier.closure = None
                    self.verifier.baseline = None
                    self.verifier.contact_seen = self.verifier.lifted = self.verifier.contact_lost = False
                if effect.phase == State.CLOSE:
                    self.verifier.closing(effect.target,goal.hand,HAND_PRESETS[goal.preset],context.sim)
                self.stop_resources = {effect.op_id: (goal.hand if name == 'hand' else 'joints',
                    tuple(str(i) for i in range(1,7)) if name == 'hand' else tuple(JOINTS))}
                self.transport.send(effect.op_id, self.actions[name], goal)
        except Exception as exc:
            self.emit(Event('REJECTED', effect.op_id, repr(exc), ok=False, stopped=True))

    def accept_scene(self, event):
        phase, response = event.data
        if not response.success:
            return Event('RESULT', event.op_id, response.message, ok=False, stopped=True)
        s = response.scene
        if phase == State.REFRESH_TARGET:
            old = self.scenes.get('table')
            target = self.refresh_target
            before = [o for o in old.objects if (o.name or o.track_id)==target] if old else []
            after = [o for o in s.objects if (o.name or o.track_id)==target]
            if len(before)!=1 or len(after)!=1:
                return Event('RESULT',event.op_id,'target_ambiguous',ok=False,stopped=True)
            a,b = before[0].center,after[0].center
            if (not all(math.isfinite(v) for v in (a.x,a.y,a.z,b.x,b.y,b.z))
                    or (a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2 > .02**2):
                return Event('RESULT',event.op_id,'target_moved',ok=False,stopped=True)
        self.scenes['table' if phase in (State.OBSERVE, State.REFRESH_TARGET) else 'destination'] = s
        stamp = s.header.stamp.sec + s.header.stamp.nanosec * 1e-9
        objects = tuple(Object(o.name or o.track_id, o.category, o.name, o.cap_color,
                               o.confidence, o.partially_visible, o.lying) for o in s.objects)
        containers = tuple(Container(x.id, x.kind, tuple(i for i, occupied in enumerate(x.slot_occupied)
                                                        if not occupied and i < len(x.slots))) for x in s.containers)
        return Event('RESULT', event.op_id, Scene(stamp, objects, containers, s.header.frame_id), stopped=True)

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def tcp(self, arm, now):
        try:
            tf = self.tf.lookup_transform('pelvis', arm+'_tcp_link', Time())
            t = stamp(tf)
            v = tf.transform.translation
            if 0 <= now-t <= .2 and all(math.isfinite(x) for x in (v.x,v.y,v.z)):
                return t,(v.x,v.y,v.z)
        except Exception:
            pass
        return None

    def stopped(self, op_id, since):
        resource = self.stop_resources.get(op_id)
        return bool(resource and self.measurements.window(resource[0],self.now(),since,resource[1]))

    def poll(self, context):
        self.transport.poll()
        waiting = self.motion_wait
        if waiting:
            self.motion_wait = None
            if context.active and context.active.id == waiting.op_id and context.state != State.CANCELLING:
                if context.sim-context.active.sim_start >= 3.0:
                    self.emit(Event('REJECTED',waiting.op_id,'motion_unavailable',ok=False,stopped=True))
                else:
                    self.execute(waiting,context)
        self.poll_scene(context)
        v = self.verifier
        if (v.contact_seen and context.target and context.hand.value == 'HOLDING'
                and context.state not in (State.RELEASE,State.RETREAT,State.VERIFY_PLACE,State.CANCELLING,State.DONE)):
            sample = self.measurements.hand(v.arm,context.sim,context.sim-.6,v.closure)
            if sample is None or sample[1] != 'HOLDING':
                v.contact_lost = True
                self.emit(Event('SENSOR',data=Evidence(context.sim,source='contact_lost_or_stale')))
        effect = self.verify_pending
        if not effect or not context.active or context.active.id != effect.op_id:
            return
        if effect.phase == State.VERIFY_PLACE:
            if self.verify_future is not None and self.verify_future.done():
                try:
                    response = self.verify_future.result()
                    if response.success:
                        self.verify_scene = response.scene
                except Exception:
                    pass
                self.verify_future = None
            if self.verify_scene is None or stamp(self.verify_scene) < context.active.sim_start:
                if self.verify_future is None and self.perceive_cli.service_is_ready():
                    self.verify_future = self.perceive_cli.call_async(Perceive.Request(camera='head',region='table'))
                if context.sim-context.active.sim_start < 3.0:
                    return
        ev = self.verifier.read(context,self.verify_scene)
        # Await fresh/stable measurements, but never wait forever for unavailable sensors.
        if ev is None and context.sim-context.active.sim_start < 3.0:
            return
        self.verify_pending = None
        self.emit(Event('RESULT',effect.op_id,ev or Evidence(context.sim,source='sensors_unconfirmed')))

    def poll_scene(self, context):
        effect = self.scene_pending
        if not effect:
            return
        if not context.active or context.active.id != effect.op_id or context.state == State.CANCELLING:
            self.scene_pending = self.scene_future = None
            return
        expired = context.sim-context.active.sim_start >= 3.0
        future = self.scene_future
        if future is not None and future.done():
            try:
                response = future.result()
                st = stamp(response.scene)
                # Camera and /clock arrive on different DDS topics. Do not reject a
                # newer camera frame just because this owner's last clock is behind it.
                if response.success and st > context.sim and not expired:
                    return
                lower = context.scene_invalidated_at
                fresh = (response.success and 0 <= context.sim-st <= context.config.freshness
                         and (lower is None or st >= lower))
                if fresh or expired:
                    self.scene_pending = self.scene_future = None
                    self.emit(Event('ROS_SCENE',effect.op_id,(effect.phase,response),ok=response.success,stopped=True))
                    return
            except Exception:
                pass
            self.scene_future = None
            self.next_scene_request = context.sim + .1
        if expired:
            self.scene_pending = self.scene_future = None
            self.emit(Event('RESULT',effect.op_id,'fresh_scene_unavailable',ok=False,stopped=True))
            return
        if self.scene_future is None and context.sim >= self.next_scene_request and self.perceive_cli.service_is_ready():
            region = 'hood' if effect.phase == State.DESTINATION and context.spec.kind == 'tubes' else 'table'
            camera = 'torso' if context.spec.kind == 'tubes' and region == 'table' else 'head'
            try:
                self.scene_future = self.perceive_cli.call_async(Perceive.Request(camera=camera,region=region))
            except Exception:
                self.next_scene_request = context.sim + .1
