"""Bounded diagnostic output; ROS RPC callbacks never wait for a log sink."""
import json
from queue import Empty, Full, Queue
from threading import Thread


class DiagnosticSink:
    def __init__(self, logger):
        self.logger = logger
        self.queue = Queue(maxsize=16)
        self.thread = Thread(target=self._run, name='perception-diagnostics', daemon=True)
        self.thread.start()

    def submit(self, event):
        try:
            self.queue.put_nowait(event)
        except Full:
            pass  # Losing diagnostics is preferable to delaying sensor input.

    def _run(self):
        while True:
            event = self.queue.get()
            if event is None:
                return
            try:
                self.logger.info(json.dumps(event, allow_nan=False))
            except Exception:
                pass

    def close(self):
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            pass
        self.queue.put_nowait(None)
        self.thread.join(timeout=.2)
