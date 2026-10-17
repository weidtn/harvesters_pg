#!/usr/bin/env python3
"""Connect/disconnect stress test for ui.py (GenTL/harvesters path).

Drives the real MainWindow headlessly through repeated connect/disconnect
cycles - the same code path a user clicking Connect/Disconnect by hand goes
through, including the settings-panel population (now guarded against the
ConnectionConfig/CxpLinkConfiguration nodes) and the shared-Harvester
device-level open/close per cycle.

Progress is written to PROGRESS_LOG, flushed and fsynced after every single
cycle, specifically so that if the machine hard-locks mid-run the last
completed cycle is still on disk after a hard reboot - stdout/whatever SSH
session launched this will not survive that, the file will.
"""

import os
import sys
import time

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

import ui as ui_mod

PROGRESS_LOG = "/tmp/stress_ui_progress.log"
MAX_CYCLES = 150
MAX_SECONDS = 20 * 60  # hard cap so a stable run does not go forever

# Never let a modal dialog block this headless run.
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
    window = ui_mod.MainWindow()
    window.populate_video_sources()

    source_index = -1
    for i in range(window.ui.cb_source.count()):
        text = window.ui.cb_source.itemText(i)
        if text != "random" and "Webcam" not in text:
            source_index = i
            break
    if source_index < 0:
        log("FATAL: no Harvester camera source found in cb_source")
        return 1
    window.ui.cb_source.setCurrentIndex(source_index)
    log(f"=== stress_ui starting: source={window.ui.cb_source.currentText()!r} "
        f"cap={MAX_CYCLES} cycles / {MAX_SECONDS}s ===")

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
