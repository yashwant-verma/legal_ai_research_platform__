"""Serialize KB mutation and experiment execution in the supported single-worker server."""
from threading import RLock
from functools import wraps
kb_lock = RLock()
def stable_kb(fn):
    @wraps(fn)
    def guarded(*args, **kwargs):
        with kb_lock:
            return fn(*args, **kwargs)
    return guarded
