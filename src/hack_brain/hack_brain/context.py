"""ROS-independent mission data. No position-derived object identities."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(str, Enum):
    WAIT = 'WAIT'
    VERIFY_EMPTY = 'VERIFY_EMPTY'
    OPEN_HANDS = 'OPEN_HANDS'
    ARMS_READY = 'ARMS_READY'
    LOOK_TABLE = 'LOOK_TABLE'
    OBSERVE = 'OBSERVE'
    OPEN = 'OPEN'
    APPROACH = 'APPROACH'
    DESCEND = 'DESCEND'
    CLOSE = 'CLOSE'
    VERIFY_GRASP = 'VERIFY_GRASP'
    LIFT = 'LIFT'
    VERIFY_LIFT = 'VERIFY_LIFT'
    REORIENT = 'REORIENT'
    VERIFY_REORIENT = 'VERIFY_REORIENT'
    DESTINATION = 'DESTINATION'
    VERIFY_TRANSFER = 'VERIFY_TRANSFER'
    TRANSFER = 'TRANSFER'
    VERIFY_HOLD = 'VERIFY_HOLD'
    LOWER = 'LOWER'
    INSERT = 'INSERT'
    RELEASE = 'RELEASE'
    VERIFY_PLACE = 'VERIFY_PLACE'
    RETREAT = 'RETREAT'
    RECOVER = 'RECOVER'
    BACK_OFF = 'BACK_OFF'
    CLEAR = 'CLEAR'
    CANCELLING = 'CANCELLING'
    DONE = 'DONE'


class Hand(str, Enum):
    EMPTY = 'EMPTY'
    HOLDING = 'HOLDING'
    UNKNOWN = 'UNKNOWN'


class Lift(str, Enum):
    NEVER_LIFTED = 'NEVER_LIFTED'
    LIFTED = 'LIFTED'
    UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True)
class Object:
    id: str
    category: str
    name: str = ''
    cap: str = ''
    confidence: float = 1.0
    partial: bool = False
    lying: bool = False
    side: str = ''  # 'left' for y > 0, else 'right': the arm when the tray does not choose it (choose_arm)


@dataclass(frozen=True)
class Container:
    id: str
    kind: str
    free_slots: tuple = ()  # indices, not inferred from absent occupancy data
    side: str = ''  # a side tray: the arm on its side carries to it (choose_arm); '' for the centre one


@dataclass(frozen=True)
class Scene:
    stamp: float
    objects: tuple = ()
    containers: tuple = ()
    frame: str = 'pelvis'


@dataclass(frozen=True)
class Evidence:
    stamp: float
    hand: Hand = Hand.UNKNOWN
    object_id: str = ''
    lifted: bool = False
    placed: bool = False
    upright: bool = False
    destination: str = ''
    slot: Optional[int] = None
    source: str = ''
    empty_arms: frozenset = frozenset()  # VERIFY_EMPTY: the hands measured empty when the other one is not


@dataclass
class History:
    attempts: int = 0
    lift_attempts: int = 0
    lift: Lift = Lift.NEVER_LIFTED
    placed: bool = False
    released: bool = False  # let go over the destination, placement not confirmed: never picked again
    contacted: bool = False


@dataclass
class Operation:
    id: int
    phase: State
    sim_start: float
    wall_start: float
    accepted: bool = False
    cancel_ack: bool = False
    terminal: bool = False
    stopped: bool = False


@dataclass(frozen=True)
class Config:
    freshness: float = 2.0
    # The first MoveTcp loads cuRobo and raises the other arm first: 15 wall s were not enough on the bench.
    operation_timeout: float = 60.0
    watchdog: float = 15.0
    cancel_timeout: float = 5.0
    cycle_estimate: float = 40.0
    reserve: float = 10.0
    max_attempts: int = 2
    max_recoveries: int = 6
    # A hand that reads neither empty nor holding is opened, both arms raised, and the mission goes on; this
    # many times per mission, then it ends.
    max_unknown_frees: int = 2
    # Perception finds 1-3 of the trays per call: an empty answer is looked at again before giving up.
    max_empty_observations: int = 2
    min_confidence: float = 0.6
    # Names left untouched: `skip: true` in grasp_db.yaml. Lifted and dropped is -10, untouched is 0 (SCORING.md).
    # Not a ROS parameter: brain_node fills it from the grasp db.
    skip_names: frozenset = frozenset()


@dataclass
class Context:
    config: Config = field(default_factory=Config)
    state: State = State.WAIT
    spec: object = None
    capabilities: frozenset = frozenset()
    sim: Optional[float] = None
    wall: float = 0.0
    last_progress_wall: float = 0.0
    started_wall: Optional[float] = None
    deadline: float = 0.0
    start_sim: Optional[float] = None
    scene_invalidated_at: Optional[float] = None
    hand: Hand = Hand.UNKNOWN
    scene: Optional[Scene] = None
    target: Optional[Object] = None
    destination: str = ''
    slot: Optional[int] = None
    reoriented: bool = False
    used_slots: set = field(default_factory=set)
    history: dict = field(default_factory=dict)
    active: Optional[Operation] = None
    serial: int = 0
    recoveries: int = 0
    empty_observations: int = 0
    # The arm may be down at the object (DESCEND was accepted and no LIFT or BACK_OFF since).
    arm_low: bool = False
    # An arm may be over the table, in the head camera view (APPROACH was accepted and no CLEAR since).
    # An arm left at the tray or above an object hid three of nine objects and all three trays from the look.
    arm_out: bool = False
    # RETREAT is done; the look that follows checks the placement (VERIFY_PLACE) instead of a new OBSERVE.
    place_look: bool = False
    # Start steps still to run, (OPEN_HANDS | ARMS_READY, arm) one per operation, and whether they already ran.
    start_steps: list = field(default_factory=list)
    start_cleared: bool = False
    disabled_arm: str = ''  # a hand that stays shut after every free: the mission goes on with the other one
    # LOOK_TABLE with the head turned to this side ('left' | 'right') to find a tray cut off by the frame; the
    # sides already looked at, once per mission (clients remember the trays a look showed).
    look_side: str = ''
    looked_sides: set = field(default_factory=set)
    unknown_frees: int = 0
    last_recovery: str = ''
    reason: str = ''
    blocked: bool = False
    cancel_wall: float = 0.0


@dataclass(frozen=True)
class Event:
    kind: str
    op_id: int = 0
    data: object = None
    ok: bool = True
    stopped: bool = False


@dataclass(frozen=True)
class Effect:
    kind: str
    op_id: int = 0
    phase: Optional[State] = None
    target: Optional[Object] = None
    destination: str = ''
    slot: Optional[int] = None
    # OPEN_HANDS / ARMS_READY: the hand to open, the arm to raise. LOOK_TABLE: the side to turn the head to.
    arm: str = ''
