"""ROS adapter. Callbacks only enqueue events; no callback waits on a future."""
from hack_interfaces.action import Hand, MoveJoints, MoveTcp
from hack_interfaces.srv import Perceive
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from .context import Container, Event, Evidence, Object, Scene, State
from .fsm import VERIFY
from .skills import Skills
from .transport import ActionTransport


class Clients:
    # No implementation currently provides verified hold/place evidence, detection,
    # IK/linear paths, insertion or reorientation. Enable capabilities only when the
    # corresponding adapter and its evidence contract have been implemented/tested.
    capabilities = frozenset()
    unavailable = ('detect and IK are stubs; linear, verified hold/place, tube skills unavailable')

    def __init__(self, node, emit, db):
        self.node, self.emit = node, emit
        cb = ReentrantCallbackGroup()
        self.perceive_cli = node.create_client(Perceive, 'perception/perceive', callback_group=cb)
        self.actions = {
            'tcp': ActionClient(node, MoveTcp, 'motion/move_tcp', callback_group=cb),
            'joints': ActionClient(node, MoveJoints, 'motion/move_joints', callback_group=cb),
            'hand': ActionClient(node, Hand, 'motion/hand', callback_group=cb),
        }
        self.transport = ActionTransport(emit)
        self.skills = Skills(db)
        self.scenes = {}

    def execute(self, effect, context):
        if effect.kind == 'CANCEL':
            if context.active.phase in VERIFY or context.active.phase in (State.OBSERVE, State.DESTINATION):
                # Read-only operations can be abandoned. Late responses retain old IDs.
                self.emit(Event('CANCEL_ACK', effect.op_id))
                self.emit(Event('RESULT', effect.op_id, ok=False, stopped=True))
            else:
                self.transport.cancel(effect.op_id)
            return
        if effect.phase in VERIFY:
            self.emit(Event('RESULT', effect.op_id, Evidence(context.sim, source='unavailable_verifier')))
            return
        if effect.phase in (State.OBSERVE, State.DESTINATION):
            if not self.perceive_cli.service_is_ready():
                self.emit(Event('REJECTED', effect.op_id, 'perception_unavailable', ok=False, stopped=True))
                return
            region = 'hood' if effect.phase == State.DESTINATION and context.spec.kind == 'tubes' else 'table'
            camera = 'torso' if context.spec.kind == 'tubes' and region == 'table' else 'head'
            try:
                future = self.perceive_cli.call_async(Perceive.Request(camera=camera, region=region))
                future.add_done_callback(lambda f: self._scene(effect, f))
            except Exception as exc:
                self.emit(Event('RESULT', effect.op_id, repr(exc), ok=False, stopped=True))
            return
        try:
            # Only owner thread updates this cache from result payloads via accept_scene.
            scene = self.scenes.get('destination' if effect.phase == State.TRANSFER else 'table')
            name, goal = self.skills.command(effect, scene, context.history[effect.target.id].attempts)
            if not self.actions[name].server_is_ready():
                self.emit(Event('REJECTED', effect.op_id, 'motion_unavailable', ok=False, stopped=True))
            else:
                self.transport.send(effect.op_id, self.actions[name], goal)
        except Exception as exc:
            self.emit(Event('REJECTED', effect.op_id, repr(exc), ok=False, stopped=True))

    def _scene(self, effect, future):
        try:
            r = future.result()
            # Wrap ROS scene until the owner consumes it; never mutate mission data here.
            self.emit(Event('ROS_SCENE', effect.op_id, (effect.phase, r), ok=r.success, stopped=True))
        except Exception as exc:
            self.emit(Event('RESULT', effect.op_id, repr(exc), ok=False, stopped=True))

    def accept_scene(self, event):
        phase, response = event.data
        if not response.success:
            return Event('RESULT', event.op_id, response.message, ok=False, stopped=True)
        s = response.scene
        self.scenes['table' if phase == State.OBSERVE else 'destination'] = s
        stamp = s.header.stamp.sec + s.header.stamp.nanosec * 1e-9
        objects = tuple(Object(o.name or o.track_id, o.category, o.name, o.cap_color,
                               o.confidence, o.partially_visible, o.lying) for o in s.objects)
        containers = tuple(Container(x.id, x.kind, tuple(i for i, occupied in enumerate(x.slot_occupied)
                                                        if not occupied and i < len(x.slots))) for x in s.containers)
        return Event('RESULT', event.op_id, Scene(stamp, objects, containers, s.header.frame_id), stopped=True)
