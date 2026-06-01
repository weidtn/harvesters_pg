import cv2
import numpy as np
import tqdm
import cam
import pyqtgraph as pg


FPS = 1000
FRAMES = 1000

ctis = cam.find_common_cti_files()
h, ia = cam.setup_cam(ctis[0],
                      fps=FPS)
timestamps = np.zeros(FRAMES)

for i in tqdm.tqdm(range(10), desc="warmup"):
    buffer = ia.fetch()
    buffer.queue()

width = ia.remote_device.node_map.Width.value
height = ia.remote_device.node_map.Height.value
frames = np.zeros([FRAMES, height, width], dtype=np.uint8)
try:
    for i in tqdm.tqdm(range(FRAMES), desc="Measuring"):
        with ia.fetch() as buffer:
            np.copyto(frames[i], buffer.payload.components[0].data.reshape(buffer.payload.components[0].height, buffer.payload.components[0].width))
            timestamps[i] = buffer.timestamp

finally:
    cam.disconnect_cam(h, ia)

timestamps = timestamps / 1e9
diff = np.gradient(timestamps)
mean = diff.mean()
std = diff.std()
print(f"Mean dt between frames: {mean} +- {std/FRAMES} [s]")
print(f"difference of mean to given fps: {1/FPS - mean}")
print(f"Min and max: {diff.min()} and {diff.max()}")

plotWidget = pg.plot()
plotWidget.plot(np.arange(FRAMES), np.gradient(timestamps))
plotWidget.setLabel("bottom", "Frame number")
plotWidget.setLabel("left", "dt [s]")

if __name__ == "__main__":
    pg.exec()
