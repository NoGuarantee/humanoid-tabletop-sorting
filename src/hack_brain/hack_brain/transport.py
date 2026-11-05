"""Asynchronous action lifecycle, also testable with ordinary Python futures.

Cancellation is a request, never a claim that physical motion has stopped.
"""
import threading
from dataclasses import dataclass
from .context import Event


@dataclass
class Pending:
    id: int
    handle: object = None
    cancel_requested: bool = False
    cancel_sent: bool = False
    terminal: bool = False


class ActionTransport:
    def __init__(self, emit):
        self.emit = emit
        self.lock = threading.RLock()
        self.pending = None

    def send(self, op_id, client, goal):
        with self.lock:
            if self.pending is not None:
                self.emit(Event('REJECTED', op_id, 'resource_busy', ok=False, stopped=True))
                return
            p = self.pending = Pending(op_id)
            try:
                future = client.send_goal_async(goal)
                future.add_done_callback(lambda f: self._accepted(p, f))
            except Exception as exc:
                # An exception does not prove the goal never reached the server.
                self.emit(Event('SEND_ERROR', op_id, repr(exc), ok=False))

    def _accepted(self, p, future):
        with self.lock:
            try:
                p.handle = future.result()
                if not p.handle.accepted:
                    p.terminal = True
                    if self.pending is p:
                        self.pending = None
                    self.emit(Event('REJECTED', p.id, 'goal_rejected', ok=False, stopped=True))
                    if p.cancel_requested:
                        self.emit(Event('CANCEL_ACK', p.id))
                    return
                self.emit(Event('ACCEPTED', p.id))
                p.handle.get_result_async().add_done_callback(lambda f: self._result(p, f))
                if p.cancel_requested:
                    self._cancel(p)
            except Exception as exc:
                self.emit(Event('SEND_ERROR', p.id, repr(exc), ok=False))

    def _result(self, p, future):
        with self.lock:
            try:
                response = future.result()
                result = response.result
                p.terminal = True
                # ROS terminal status is not measured physical stop. Keep cancelled
                # resources latched even if the server reports SUCCEEDED/CANCELED.
                if (not p.cancel_requested and response.status == 4
                        and bool(result.success) and self.pending is p):
                    self.pending = None
                self.emit(Event('RESULT', p.id, result,
                                ok=response.status == 4 and bool(result.success), stopped=False))
            except Exception as exc:
                self.emit(Event('SEND_ERROR', p.id, repr(exc), ok=False))

    def cancel(self, op_id):
        with self.lock:
            p = self.pending
            if p is None or p.id != op_id:
                # A result may have arrived before the cancellation effect is executed.
                # It says nothing about physical stop; do not synthesize confirmation.
                self.emit(Event('CANCEL_ACK', op_id, ok=False))
                return
            p.cancel_requested = True
            if p.handle is not None:
                self._cancel(p)

    def _cancel(self, p):
        if p.cancel_sent:
            return
        p.cancel_sent = True
        try:
            p.handle.cancel_goal_async().add_done_callback(lambda f: self._cancel_result(p, f))
        except Exception:
            self.emit(Event('CANCEL_ACK', p.id, ok=False))

    def _cancel_result(self, p, future):
        try:
            accepted = bool(future.result().goals_canceling)
        except Exception:
            accepted = False
        self.emit(Event('CANCEL_ACK', p.id, ok=accepted))
