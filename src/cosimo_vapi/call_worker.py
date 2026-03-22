"""Subprocess worker that runs a single Vapi call then exits."""

import os
import sys
import threading
import time


def run_call(public_key: str, assistant_id: str):
    from vapi_python import Vapi

    vapi = Vapi(api_key=public_key)
    print("COSIMO:CALL_START", flush=True)
    vapi.start(assistant_id=assistant_id)

    # Monitor for meeting end via OS-level stderr
    ended = threading.Event()

    read_fd, write_fd = os.pipe()
    original_fd = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def monitor():
        buf = b""
        while not ended.is_set():
            try:
                data = os.read(read_fd, 4096)
            except OSError:
                break
            if not data:
                break
            os.write(original_fd, data)
            buf += data
            if b"Meeting has ended" in buf:
                ended.set()
                break
            if len(buf) > 8192:
                buf = buf[-4096:]

    threading.Thread(target=monitor, daemon=True).start()
    ended.wait(timeout=660)

    # Don't try vapi.stop() — it hangs. Just force-exit the process.
    print("COSIMO:CALL_END", flush=True)
    os._exit(0)


if __name__ == "__main__":
    run_call(sys.argv[1], sys.argv[2])