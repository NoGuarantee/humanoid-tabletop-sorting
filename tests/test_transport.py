import unittest
from concurrent.futures import Future
from types import SimpleNamespace as NS
from hack_brain.transport import ActionTransport, STOP_WAIT_S


class Client:
    def __init__(self):
        self.future = Future()
        self.calls = 0

    def send_goal_async(self, goal):
        self.calls += 1
        return self.future


class Handle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result, self.cancel = Future(), Future()
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.t = ActionTransport(self.events.append)
        self.client = Client()

    def test_late_acceptance_is_cancelled_once(self):
        self.t.send(1, self.client, object())
        self.t.cancel(1)
        self.t.cancel(1)
        h = Handle()
        self.client.future.set_result(h)
        self.assertEqual(h.cancel_calls, 1)
        h.cancel.set_result(NS(goals_canceling=[1]))
        h.result.set_result(NS(status=5, result=NS(success=False)))
        self.assertEqual([e.kind for e in self.events], ['ACCEPTED', 'CANCEL_ACK', 'RESULT'])
        self.assertFalse(self.events[-1].stopped)
        self.assertIsNotNone(self.t.pending)
        self.t.send(2, Client(), object())
        self.assertEqual(self.events[-1].data, 'resource_busy')

    def test_one_goal_until_terminal(self):
        self.t.send(1, self.client, object())
        other = Client()
        self.t.send(2, other, object())
        self.assertEqual(other.calls, 0)
        h = Handle()
        self.client.future.set_result(h)
        self.assertEqual(self.events[-1].kind, 'ACCEPTED')
        h.result.set_result(NS(status=4, result=NS(success=True)))
        self.assertTrue(self.events[-1].ok)
        self.t.send(3, other, object())
        self.assertEqual(other.calls, 1)

    def test_late_rejection_after_cancel(self):
        self.t.send(1, self.client, object())
        self.t.cancel(1)
        self.client.future.set_result(Handle(accepted=False))
        self.assertEqual([e.kind for e in self.events], ['REJECTED', 'CANCEL_ACK'])
        self.assertTrue(self.events[0].stopped)

    def test_send_future_exception_keeps_resource(self):
        self.t.send(1, self.client, object())
        self.client.future.set_exception(RuntimeError('connection lost'))
        self.assertEqual(self.events[-1].kind, 'SEND_ERROR')
        self.assertIsNotNone(self.t.pending)

    def test_result_exception_keeps_handle_for_cancel(self):
        self.t.send(1, self.client, object())
        h = Handle()
        self.client.future.set_result(h)
        h.result.set_exception(RuntimeError('result lost'))
        self.t.cancel(1)
        self.assertEqual(h.cancel_calls, 1)

    def test_failed_result_retains_handle_for_cancel(self):
        self.t.send(1, self.client, object())
        h = Handle()
        self.client.future.set_result(h)
        h.result.set_result(NS(status=4, result=NS(success=False)))
        self.assertFalse(self.events[-1].stopped)
        self.assertIsNotNone(self.t.pending)
        self.t.send(2, Client(), object())
        self.assertEqual(self.events[-1].data, 'resource_busy')
        self.t.cancel(1)
        self.assertEqual(h.cancel_calls, 1)

    def test_cancel_refused(self):
        self.t.send(1, self.client, object())
        h = Handle()
        self.client.future.set_result(h)
        self.t.cancel(1)
        h.cancel.set_result(NS(goals_canceling=[]))
        self.assertFalse(self.events[-1].ok)
        self.assertIsNotNone(self.t.pending)


if __name__ == '__main__':
    unittest.main()

class MeasuredTransportTests(unittest.TestCase):
    def setUp(self):
        self.time=10.;self.rest=False;self.events=[]
        self.t=ActionTransport(self.events.append,lambda op,since:self.rest,lambda:self.time)
        self.c=Client();self.h=Handle()
        self.t.send(1,self.c,object());self.c.future.set_result(self.h)

    def test_failed_terminal_waits_for_measured_stop_then_releases_resource(self):
        self.h.result.set_result(NS(status=4,result=NS(success=False)))
        self.t.poll();self.assertEqual(self.events[-1].kind,'ACCEPTED')
        self.rest=True;self.t.poll()
        self.assertTrue(self.events[-1].stopped);self.assertFalse(self.events[-1].ok)
        self.assertIsNone(self.t.pending)

    def test_success_also_requires_settling(self):
        self.h.result.set_result(NS(status=4,result=NS(success=True)))
        self.time=10+STOP_WAIT_S+1;self.t.poll()
        self.assertFalse(self.events[-1].ok);self.assertFalse(self.events[-1].stopped)
        self.assertIsNotNone(self.t.pending)

    def test_cancel_terminal_goal_does_not_send_invalid_ros_cancel(self):
        self.h.result.set_result(NS(status=4,result=NS(success=False)))
        self.t.cancel(1)
        self.assertEqual(self.h.cancel_calls,0)
        self.assertEqual(self.events[-1].kind,'CANCEL_ACK')
        self.rest=True;self.t.poll()
        self.assertTrue(self.events[-1].stopped)

    def test_timeout_racing_measured_result_can_ack_completed_operation(self):
        self.h.result.set_result(NS(status=4,result=NS(success=True)))
        self.rest=True;self.t.poll();self.t.cancel(1)
        self.assertEqual(self.events[-1].kind,'CANCEL_ACK');self.assertTrue(self.events[-1].ok)
