import os
import sys
from contextlib import contextmanager


class Tee:
    """Mirror a stream to a file so console output is also saved to disk.

    A `Tee` is a minimal file-like object: it implements `write`, `flush` and `isatty`,
    which is all `print`, the traceback machinery and the logging handlers call. Every
    write is forwarded to both the original stream (the console) and a file, and the file
    is flushed each time so the log is tailable live and survives an abrupt exit.
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

    def isatty(self):
        # Always False, and it must exist: `logging` handlers ask the stream whether it is a
        # terminal before emitting ANSI colour, and a missing method there is not an error the
        # caller sees — `logging` catches it, prints "--- Logging error ---" and DROPS the record.
        # Optuna logs each trial's outcome through such a handler, so without this the search log
        # loses its entire per-trial history while the search itself runs on unaffected. False
        # rather than delegating to the wrapped stream: a Tee always writes to a file as well, and
        # colour escapes belong in neither the file nor the archived record.
        return False


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
