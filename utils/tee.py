import os
import sys
from contextlib import contextmanager


class Tee:
    """Mirror a stream to a file so console output is also saved to disk.

    A `Tee` is a minimal file-like object: it only implements `write` and `flush`,
    which is all `print` and traceback machinery call. Every write is forwarded to
    both the original stream (the console) and a file, and the file is flushed each
    time so the log is tailable live and survives an abrupt exit.
    """

    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


@contextmanager
def tee_to_file(path):
    """Within the context, mirror stdout and stderr to `path` (and still the console).

    A `path` of `None` disables file logging and yields without side effects, so call
    sites can pass an optional path directly. The parent directory is created if
    needed, and the original streams are restored on exit even if the body raises.
    """
    if path is None:
        yield
        return

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    file = open(path, 'w')
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(stdout, file)
    sys.stderr = Tee(stderr, file)
    try:
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
        file.close()
