#!/usr/bin/env python3
"""Connect/disconnect stress test for phx_ui.py (raw PHX C API path).

Same shape as stress_ui.py, driving PhxMainWindow's connect_video()/
disconnect_video() in a loop instead - each cycle is: one short GenICam
session to configure the camera (fully torn down), then raw PHX
PHX_Create/PHX_Open, then PHX_Close/PHX_Destroy on disconnect. No GenTL
producer traffic happens except the brief configure step.

Progress is written to PROGRESS_LOG, flushed and fsynced after every single
cycle, so the last completed cycle survives a hard reboot even if this
process does not.
"""

import os
import sys
import time

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

import phx_ui

PROGRESS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_phx_progress.log")  # /tmp is tmpfs here and does not survive a hard reboot
MAX_CYCLES = 150
MAX_SECONDS = 20 * 60

QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: None)
QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: None)
QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: None)

_logf = open(PROGRESS_LOG, "a", buffering=1)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()
    os.fsync(_logf.fileno())


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = phx_ui.PhxMainWindow()
    log(f"=== stress_phx starting: cap={MAX_CYCLES} cycles / {MAX_SECONDS}s ===")

    t_start = time.monotonic()
    cycle = 0
    while cycle < MAX_CYCLES and (time.monotonic() - t_start) < MAX_SECONDS:
        cycle += 1
        t0 = time.monotonic()
        try:
            window.connect_video()
            if not window.videothread:
                log(f"cycle {cycle}: CONNECT FAILED (no exception, no videothread) - stopping")
                break
            app.processEvents()
            window.disconnect_video()
            app.processEvents()
        except Exception as e:
            log(f"cycle {cycle}: EXCEPTION {type(e).__name__}: {e}")
            raise
        log(f"cycle {cycle}: OK ({time.monotonic() - t0:.2f}s)")
        time.sleep(0.3)

    if cycle >= MAX_CYCLES:
        log(f"=== COMPLETED {cycle} cycles (cap reached) without a crash ===")
    else:
        log(f"=== STOPPED after {cycle} cycles ({time.monotonic() - t_start:.1f}s elapsed) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
