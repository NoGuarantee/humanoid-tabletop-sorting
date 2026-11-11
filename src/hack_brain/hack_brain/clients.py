"""ROS adapter. Callbacks only enqueue events; no callback waits on a future."""
import math
import json

from hack_interfaces.action import Hand, MoveJoints, MoveTcp
from hack_interfaces.srv import Perceive
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger

from .context import Container, Event, Evidence, Object, Scene, State
from .physical import PhysicalVerifier
from .sensors import HAND_STILL, Measurements, stamp, with_known_trays

# Joint spread over 0.3 s that still counts as the arm at rest after a move. The drives sway 0.005..0.016 rad
# while they settle; at 0.004 one sway outlasted the 4 s wait after a CLEAR that had arrived and ended the mission
# after the first sorted object (eval 2026-11-08). A driven joint at a tenth of the 0.8 rad/s limit moves 0.024.
JOINT_STILL = 0.02
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from hack_motion.robot import HAND_PRESETS, JOINTS
from .fsm import VERIFY
from .skills import Skills
from .recorded_grasps import observed_pose, associate_model_pose, selected_object
from .grasp_planner import tray_side
from .transport import ActionTransport


SCENE_WAIT_S = 6.0  # sim s an OBSERVE or DESTINATION waits for a fresh scene: the first look after an arm
                   # move took 4-6 s wall, 2.5-3.7 s sim, waiting for /tf (eval 2026-11-08)
# sim s a TCP pose from /tf may lag /clock. 0.2 s left the grasp baseline empty: every lift of the held cartridge
# read as not lifted and the mission ended lift_unconfirmed (eval 2026-11-08). Lift checks also require a sample
# stamped after the phase start, so an old pose cannot pass for the new one.
TCP_MAX_AGE = 2.0
UNSURE_S = 1.5  # sim s without a HOLDING reading before a held object counts as lost


def stop_resource(name, goal):
    """(measurement key, joint names) that must read at rest after the goal.

    A MoveTcp settles its own arm, the waist and the head. The other arm it only holds within 0.05 rad: after a
    left descend the right shoulder and elbow kept moving 0.02 rad per 0.4 s for over 4 s, the stop over every
    joint never came, and the mission ended (eval 2026-11-08)."""
    if name == 'hand':
        return goal.hand, tuple(str(i) for i in range(1,7))
    if name == 'tcp':
        other = '_r_joint' if goal.arm == 'left' else '_l_joint'
        return 'joints', tuple(n for n in JOINTS if not n.endswith(other))
    return 'joints', tuple(JOINTS)


class Clients:
    # Sorting evidence is produced by PhysicalVerifier, never by action success.
    # Tube skills and rack verification remain unavailable.
    capabilities = frozenset({'perception', 'pick', 'transfer', 'release', 'retreat', 'place', 'verification'})
    unavailable = 'sorting sensor verification enabled; tube skills unavailable'

    def __init__(self, node, emit, db):
        self.node, self.emit = node, emit
        cb = ReentrantCallbackGroup()
        self.perceive_cli = node.create_client(Perceive, 'perception/perceive', callback_group=cb)
        self.model_cli = node.create_client(Trigger, 'perception/estimate_model_poses', callback_group=cb)
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
        # /tf on its own node and thread. On this node's executor every /tf message became a task in the shared
        # thread pool (tf2_ros puts it in a reentrant group): /tf lagged 1.2 s, and under load /clock and the camera
        # callbacks waited behind it for over 10 s (eval 2026-11-08).
        self.tf_listener = TransformListener(self.tf, None, spin_thread=True)
        self.verifier = PhysicalVerifier(self.measurements,self.tcp)
        self.stop_resources = {}
        self.stop_reported = None
        self.scene_pending = None
        self.scene_future = None
        self.next_scene_request = 0.0
        # The last Perceive call and its (camera, region), kept when its phase ends. Perception serves calls on four
        # threads: a phase that abandoned a slow call and sent another stacked them until /tf and /clock starved
        # there and every call took 8-14 s (eval 2026-11-08). A new phase takes the call over instead.
        self.perceive_inflight = None
        self.motion_wait = None
        self.model_pending = None
        self.model_future = None
        self.next_model_request = 0.
        self.model_pose_error = ''
        self.verify_pending = None
        self.verify_scene = None
        self.verify_future = None
        self.transport = ActionTransport(emit,self.stopped,self.now)
        self.skills = Skills(db)
        self.scenes = {}
        self.trays = {}  # tray id -> the last hack_interfaces Container seen, see with_known_trays

    def execute(self, effect, context):
        if effect.kind == 'CANCEL':
            if self.model_pending is not None and self.model_pending.op_id == effect.op_id:
                self.model_pending = self.model_future = None
                self.emit(Event('CANCEL_ACK', effect.op_id))
                self.emit(Event('RESULT', effect.op_id, ok=False, stopped=True))
                return
            if self.motion_wait is not None and self.motion_wait[0].op_id == effect.op_id:
                self.motion_wait = None
                self.emit(Event('CANCEL_ACK',effect.op_id))
                self.emit(Event('RESULT',effect.op_id,ok=False,stopped=True))
                return
            if context.active.phase in VERIFY or context.active.phase in (State.OBSERVE, State.DESTINATION):
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
            self.verify_future = self.adopt_perceive(('head', 'table'))
            return
        if effect.phase in (State.OBSERVE, State.DESTINATION):
            self.scene_pending = effect
            self.scene_future = self.adopt_perceive(self.scene_request(effect.phase, context))
            self.next_scene_request = 0.0
            return
        if self.skills.needs_model_pose(effect):
            self.model_pending = effect
            self.model_pose_error = ''
            self.model_future = None
            self.next_model_request = 0.
            return
        self.execute_motion(effect, context)

    def execute_motion(self, effect, context):
        try:
            # Only owner thread updates this cache from result payloads via accept_scene.
            scene = self.scenes.get('destination' if effect.phase == State.TRANSFER else 'table')
            name, goal = self.skills.command(effect, scene, context.history[effect.target.id].attempts if effect.target else 0)
            self.send_motion(effect, name, goal, context.sim)
        except Exception as exc:
            self.emit(Event('REJECTED', effect.op_id, repr(exc), ok=False, stopped=True))

    def send_motion(self, effect, name, goal, now):
        if self.skills.needs_model_pose(effect):
            exposure = self.skills.grasp.exposure
            if not 0 <= now - exposure['stamp_ns'] / 1e9 <= 1.75:
                self.motion_wait = None
                self.emit(Event('REJECTED', effect.op_id, 'recorded_pose_expired_before_motion', ok=False, stopped=True))
                return
        if not self.actions[name].server_is_ready():
            self.motion_wait = (effect, name, goal)
            return
        self.motion_wait = None
        if effect.phase == State.OPEN:
            self.verifier.arm = goal.hand
            self.verifier.target_id = effect.target.id
            self.verifier.closure = None
            self.verifier.baseline = None
            self.verifier.contact_seen = self.verifier.lifted = self.verifier.contact_lost = False
            self.verifier.unsure_since = None
        if effect.phase == State.CLOSE:
            self.verifier.closing(effect.target, goal.hand, HAND_PRESETS[goal.preset], now)
        if effect.phase == State.TIGHTEN:
            self.verifier.closure = tuple(HAND_PRESETS[goal.preset])
            self.verifier.closures[goal.hand] = self.verifier.closure
            self.verifier.unsure_since = None
        self.stop_resources = {effect.op_id: stop_resource(name, goal)}
        self.transport.send(effect.op_id, self.actions[name], goal)

    def accept_scene(self, event):
        phase, response = event.data
        if not response.success:
            return Event('RESULT', event.op_id, response.message, ok=False, stopped=True)
        s = with_known_trays(response.scene, self.trays)
        self.scenes['table' if phase == State.OBSERVE else 'destination'] = s
        stamp = s.header.stamp.sec + s.header.stamp.nanosec * 1e-9
        objects = tuple(Object(o.name or o.track_id, o.category, o.name, o.cap_color,
                               o.confidence, o.partially_visible, o.lying,
                               'left' if o.center.y > 0 else 'right', o.track_id) for o in s.objects)
        containers = tuple(Container(x.id, x.kind, tuple(i for i, occupied in enumerate(x.slot_occupied)
                                                        if not occupied and i < len(x.slots)),
                                     tray_side(x)) for x in s.containers)
        return Event('RESULT', event.op_id, Scene(stamp, objects, containers, s.header.frame_id), stopped=True)

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def tcp(self, arm, now):
        try:
            tf = self.tf.lookup_transform('pelvis', arm+'_tcp_link', Time())
            t = stamp(tf)
            v = tf.transform.translation
            if 0 <= now-t <= TCP_MAX_AGE and all(math.isfinite(x) for x in (v.x,v.y,v.z)):
                return t,(v.x,v.y,v.z)
        except Exception:
            pass
        return None

    def stopped(self, op_id, since):
        resource = self.stop_resources.get(op_id)
        # A squeezing hand jitters (sensors.HAND_STILL); the joint tolerance never saw it settle.
        tolerance = HAND_STILL if resource and resource[0] in ('left', 'right') else JOINT_STILL
        now = self.now()
        still = bool(resource and self.measurements.window(resource[0],now,since,resource[1],tolerance=tolerance))
        if resource and not still and now - since >= 1.0 and self.stop_reported != op_id:
            self.stop_reported = op_id
            self.node.get_logger().warning(f"op {op_id}: no measured stop 1 s after the result. "
                + self.measurements.still_report(resource[0],now,since,resource[1],tolerance=tolerance))
        return still

    def poll(self, context):
        self.transport.poll()
        self.poll_model_pose(context)
        waiting = self.motion_wait
        if waiting:
            self.motion_wait = None
            effect, name, goal = waiting
            if context.active and context.active.id == effect.op_id and context.state != State.CANCELLING:
                if context.sim-context.active.sim_start >= 3.0:
                    self.emit(Event('REJECTED',effect.op_id,'motion_unavailable',ok=False,stopped=True))
                else:
                    self.send_motion(effect, name, goal, context.sim)
        self.poll_scene(context)
        v = self.verifier
        if (v.contact_seen and context.target and context.hand.value == 'HOLDING'
                and context.state not in (State.TIGHTEN,State.RELEASE,State.RETREAT,State.VERIFY_PLACE,State.CANCELLING,State.DONE)):
            sample = self.measurements.hand(v.arm,context.sim,context.sim-.6,v.closure)
            if sample is not None and sample[1] == 'HOLDING':
                v.unsure_since = None
            elif v.unsure_since is None and (sample is None or sample[1] != 'EMPTY'):
                # Fingers shift under the load while the arm lifts or carries: a reading that is not still
                # for 0.3 s is not a loss. A hand closed to its target is, and so is no HOLDING for UNSURE_S.
                v.unsure_since = context.sim
            elif (sample is not None and sample[1] == 'EMPTY') or context.sim-v.unsure_since >= UNSURE_S:
                v.contact_lost = True
                rows = list(self.measurements.samples[v.arm])[-4:]
                self.node.get_logger().warn(
                    f"hold lost on {v.arm}: {'closed to target' if sample else 'no HOLDING for %.1f s' % UNSURE_S}, "
                    f"target {v.closure}, last readings "
                    + '; '.join(f"{t:.2f}: " + ' '.join(f"{values.get(str(i), -1):.2f}" for i in range(1, 7))
                                + ' effort ' + ' '.join(f"{e:.2f}" for e in msg.effort) for t, values, msg in rows))
                self.emit(Event('SENSOR',data=Evidence(context.sim,source='contact_lost_or_stale')))
        effect = self.verify_pending
        if not effect or not context.active or context.active.id != effect.op_id:
            return
        if effect.phase == State.VERIFY_PLACE:
            if self.verify_future is not None and self.verify_future.done():
                try:
                    response = self.verify_future.result()
                    if response.success:
                        self.verify_scene = with_known_trays(response.scene, self.trays)
                except Exception:
                    pass
                self.verify_future = None
            if self.verify_scene is None or stamp(self.verify_scene) < context.active.sim_start:
                if self.verify_future is None and self.perceive_cli.service_is_ready():
                    self.verify_future = self.call_perceive(('head', 'table'))
                if context.sim-context.active.sim_start < SCENE_WAIT_S:
                    return
        ev = self.verifier.read(context,self.verify_scene)
        # Await fresh/stable measurements, but never wait forever for unavailable sensors.
        if ev is None and context.sim-context.active.sim_start < 3.0:
            return
        self.verify_pending = None
        self.emit(Event('RESULT',effect.op_id,ev or Evidence(context.sim,source='sensors_unconfirmed')))

    def poll_model_pose(self, context):
        """Read-only preparation of OPEN/DESCEND, owned by the same operation ID as the action.

        No future callback sends a motor command. Cancellation or a changed operation drops the response.
        Perception's Trigger schedules fitting and can return 'pending', so retry within the scene budget.
        """
        effect = self.model_pending
        if effect is None:
            return
        if not context.active or context.active.id != effect.op_id or context.state == State.CANCELLING:
            self.model_pending = self.model_future = None
            return
        # A freshly started CPU worker can still be loading at the normal
        # six-second scene deadline (4090 seed 7: ready just after it). Allow
        # bounded initialization before OPEN, without relaxing exposure age or
        # waiting longer with the hand already at DESCEND.
        budget = 15. if effect.phase == State.OPEN and self.model_pose_error == 'worker_loading' else SCENE_WAIT_S
        if context.sim - context.active.sim_start >= budget:
            self.model_pending = self.model_future = None
            self.node.get_logger().warning('recorded pose unavailable: ' + json.dumps({
                'object': effect.target.name, 'phase': effect.phase.value,
                'last_response': self.model_pose_error}))
            self.emit(Event('REJECTED', effect.op_id, 'fresh_recorded_pose_unavailable', ok=False, stopped=True))
            return
        if self.model_future is not None and self.model_future.done():
            future, self.model_future = self.model_future, None
            data = None
            try:
                response = future.result()
                payload = json.loads(response.message)
                if response.success:
                    if effect.phase == State.OPEN:
                        scene = self.scenes['table']
                        if scene.header.frame_id != 'pelvis' or not 0 <= context.active.sim_start-stamp(scene) <= 2.:
                            raise ValueError('stale_recorded_selection_scene')
                        obj = selected_object(scene, effect.target)
                        reference = [getattr(obj.center, axis) for axis in 'xyz']
                        reference_stamp = int(stamp(scene)*1e9)
                    else:
                        reference = self.skills.grasp.observed_center
                        reference_stamp = self.skills.grasp.exposure['stamp_ns']
                    data = associate_model_pose(payload, reference, reference_stamp, object_name=effect.target.name)
                    observed_pose(data, context.sim, int(context.active.sim_start * 1e9), object_name=effect.target.name)
                elif payload.get('status') != 'awaiting_fresh_pose' or not self.model_pose_error:
                    # A later pending poll must not erase the actual pose rejection.
                    self.model_pose_error = payload.get('status', 'unsuccessful_response')
            except Exception as exc:
                self.model_pose_error = str(exc)
                data = None
            if data is not None:
                self.model_pending = None
                try:
                    self.skills.prepare_recorded(effect, self.scenes['table'], data, context.sim,
                                                 int(context.active.sim_start * 1e9),
                                                 context.history[effect.target.id].attempts, context.disabled_arm)
                    self.node.get_logger().info('recorded grasp: ' + json.dumps({
                        'object': effect.target.name, 'profile': self.skills.grasp.profile_key,
                        'arm': self.skills.grasp.arm, 'phase': effect.phase.value,
                        'destination': effect.destination, 'exposure': self.skills.grasp.exposure,
                        'association': data.get('target_association'),
                        'model_pose': next((row for row in data.get('objects', [])
                                            if row.get('name') == effect.target.name), None)}))
                    self.execute_motion(effect, context)
                except Exception as exc:
                    self.emit(Event('REJECTED', effect.op_id, repr(exc), ok=False, stopped=True))
                return
            self.next_model_request = context.sim + .1
        if (self.model_future is None and context.sim >= self.next_model_request
                and self.model_cli.service_is_ready()):
            try:
                self.model_future = self.model_cli.call_async(Trigger.Request())
            except Exception:
                self.next_model_request = context.sim + .1

    @staticmethod
    def scene_request(phase, context):
        region = 'hood' if phase == State.DESTINATION and context.spec.kind == 'tubes' else 'table'
        camera = 'torso' if context.spec.kind == 'tubes' and region == 'table' else 'head'
        return camera, region

    def call_perceive(self, key):
        future = self.perceive_cli.call_async(Perceive.Request(camera=key[0], region=key[1]))
        self.perceive_inflight = (key, future)
        return future

    def adopt_perceive(self, key):
        """The call still in flight for the same camera and region, if any. Its scene passes the phase's
        freshness check or is dropped, as any other response."""
        if self.perceive_inflight is None:
            return None
        k, future = self.perceive_inflight
        return future if k == key and not future.done() else None

    def poll_scene(self, context):
        effect = self.scene_pending
        if not effect:
            return
        if not context.active or context.active.id != effect.op_id or context.state == State.CANCELLING:
            self.scene_pending = self.scene_future = None
            return
        expired = context.sim-context.active.sim_start >= SCENE_WAIT_S
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
            try:
                self.scene_future = self.call_perceive(self.scene_request(effect.phase, context))
            except Exception:
                self.next_scene_request = context.sim + .1
