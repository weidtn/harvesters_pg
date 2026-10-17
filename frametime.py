import time

import numpy as np
import pyqtgraph as pg
import tqdm

import cam


FPS = 1000
FRAMES = 1000


def fetch(ia):
    """Fetch one buffer, retrying until cam.FETCH_TIMEOUT_S has elapsed.

    try_fetch returns None on timeout and immediately on an incomplete buffer,
    so a single None is not a timeout.
    """
    deadline = time.monotonic() + cam.FETCH_TIMEOUT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"no frame within {cam.FETCH_TIMEOUT_S} s")
        buffer = ia.try_fetch(timeout=remaining)
        if buffer is not None:
            return buffer


def buffer_timestamp_ns(buffer) -> int:
    try:
        return buffer.timestamp_ns
    except Exception:
        return buffer.timestamp


ctis = cam.load_cti_envvar()
h, ia = cam.setup_cam(ctis, fps=FPS)
timestamps = np.zeros(FRAMES)
try:
    for i in tqdm.tqdm(range(10), desc="warmup"):
        with fetch(ia):
            pass

    width = ia.remote_device.node_map.Width.value
    height = ia.remote_device.node_map.Height.value
    frames = np.zeros([FRAMES, height, width], dtype=np.uint8)
    for i in tqdm.tqdm(range(FRAMES), desc="Measuring"):
        with fetch(ia) as buffer:
            frames[i] = cam.component_to_grayscale8(buffer.payload.components[0])
            timestamps[i] = buffer_timestamp_ns(buffer)
finally:
    cam.teardown_cam(h, ia)

timestamps = timestamps / 1e9
diff = np.gradient(timestamps)
mean = diff.mean()
std = diff.std()
print(f"Mean dt between frames: {mean} +- {std/np.sqrt(FRAMES)} [s]")
print(f"difference of mean to given fps: {1/FPS - mean}")
print(f"Min and max: {diff.min()} and {diff.max()}")

plotWidget = pg.plot()
plotWidget.plot(np.arange(FRAMES), diff)
plotWidget.setLabel("bottom", "Frame number")
plotWidget.setLabel("left", "dt [s]")

if __name__ == "__main__":
    pg.exec()
