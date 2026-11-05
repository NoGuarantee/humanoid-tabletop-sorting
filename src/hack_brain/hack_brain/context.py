"""ROS-independent mission data. No position-derived object identities."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(str, Enum):
    WAIT = 'WAIT'
    VERIFY_EMPTY = 'VERIFY_EMPTY'
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


@dataclass(frozen=True)
class Container:
    id: str
    kind: str
    free_slots: tuple = ()  # indices, not inferred from absent occupancy data


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


@dataclass
class History:
    attempts: int = 0
    lift: Lift = Lift.NEVER_LIFTED
    placed: bool = False
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
    operation_timeout: float = 20.0
    watchdog: float = 15.0
    cancel_timeout: float = 5.0
    cycle_estimate: float = 40.0
    reserve: float = 5.0
    max_attempts: int = 2
    max_recoveries: int = 6
    min_confidence: float = 0.6


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
