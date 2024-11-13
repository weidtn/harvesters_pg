import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import cam

h, ia = cam.setup_cam()

FRAMES = 1000
timestamps = np.zeros(FRAMES)

try:
    for i in tqdm.tqdm(range(FRAMES)):
        with ia.fetch() as buffer:
            timestamps[i] = buffer.timestamp

finally:
    cam.disconnect_cam(h, ia)

print(np.diff(timestamps).mean())
print(timestamps[0])
