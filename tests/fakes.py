"""Explicit test evidence, never imported by the runtime ROS adapter."""
from dataclasses import replace
from hack_brain.context import Context, Event, Evidence, Hand, Object, Container, Scene, State
from hack_brain.fsm import transition
from hack_brain.policies import required_capabilities
from hack_brain.task import TaskSpec


def sort_spec():
    return TaskSpec('sort2', 'sort', categories={'figure': 'blue', 'dish': 'red'},
                    objects={str(i): 'figure' for i in range(6)})


def tubes_spec(variant):
    return TaskSpec('tubes', 'tubes', variant,
                    {'easy': 'any_slot', 'standard': 'by_type', 'lying': 'by_type', 'color': 'by_cap_color'}[variant],
                    tubes={str(i): {} for i in range(9)},
                    racks={'hood50': {'where': 'hood', 'tube': 'falcon50', 'color': 'red'},
                           'hood15': {'where': 'hood', 'tube': 'falcon15', 'color': 'blue'},
                           'table50': {'where': 'table', 'tube': 'falcon50', 'color': 'red'}})


class Fake:
    def __init__(self, spec=None, objects=None, capabilities=None, config=None):
        self.spec = spec or sort_spec()
        self.objects = objects if objects is not None else [Object('a', 'figure', name='a'), Object('b', 'dish', name='b')]
        self.c = Context() if config is None else Context(config=config)
        self.sim, self.wall = 1.0, 100.0
        self.effects, self.commands, self.events = [], [], []
        self.send(Event('START', data=(self.spec, frozenset(required_capabilities(self.spec) or ())
                                      if capabilities is None else capabilities)))
        self.tick()

    def send(self, event):
        before = self.c
        self.c, effects = transition(self.c, event)
        self.events.append(event)
        self.effects.extend(effects)
        for effect in effects:
            if effect.kind == 'RUN':
                assert before.state != State.DONE
                assert before.active is None or event.op_id == before.active.id
                self.commands.append(effect)
        return effects

    def tick(self, sim=None, wall=None):
        if sim is not None:
            self.sim = sim
        if wall is not None:
            self.wall = wall
        return self.send(Event('TICK', data=(self.sim, self.wall)))

    def answer(self, data=None, ok=True, stopped=True):
        op = self.c.active
        return self.send(Event('RESULT', op.id, self.default() if data is None else data, ok, stopped))

    def default(self):
        c = self.c
        if c.state in (State.OBSERVE, State.DESTINATION, State.REFRESH_TARGET):
            objects = tuple(o for o in self.objects if not c.history.get(o.id) or not c.history[o.id].placed)
            containers = (Container('blue', 'tray'), Container('red', 'tray')) if self.spec.kind == 'sort' else (
                Container('hood50', 'rack', (0, 1, 2, 3)), Container('hood15', 'rack', (0, 1, 2, 3)))
            return Scene(self.sim, objects, containers)
        hand = Hand.EMPTY if c.state in (State.VERIFY_EMPTY, State.VERIFY_PLACE, State.RECOVER) else Hand.HOLDING
        return Evidence(self.sim, hand, c.target.id if c.target else '', lifted=True, placed=True,
                        upright=True, destination=c.destination, slot=c.slot, source='fake')

    def until(self, state, limit=150):
        for _ in range(limit):
            if self.c.state == state:
                return
            assert self.c.state != State.DONE, self.c.reason
            self.answer()
        raise AssertionError('unbounded mission')

    def run(self):
        self.until(State.DONE)
        return self.c
