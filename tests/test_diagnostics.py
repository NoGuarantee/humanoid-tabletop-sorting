"""A blocked log sink must not delay perception input or RPC callbacks."""
import threading
import time
import unittest
from types import SimpleNamespace
from hack_perception.diagnostics import DiagnosticSink


class DiagnosticsTest(unittest.TestCase):
    def test_full_queue_and_blocked_logger_never_block_producers(self):
        entered, release = threading.Event(), threading.Event()
        def write(message):
            entered.set()
            release.wait(timeout=2)
        sink = DiagnosticSink(SimpleNamespace(info=write))
        try:
            sink.submit({'event': 'start'})
            self.assertTrue(entered.wait(timeout=1))
            started = time.monotonic()
            for i in range(100):
                sink.submit({'sequence': i})
            self.assertLess(time.monotonic()-started, .1)
            self.assertLessEqual(sink.queue.qsize(), 16)
        finally:
            release.set()
            sink.close()
