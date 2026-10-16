#!/usr/bin/env python3
import cam
import numpy as np
import h5py
import tqdm
import datetime
import uuid

from harvesters.core import Harvester, ImageAcquirer

FPS = 1000
T = 1


ctis = cam.find_common_cti_files()
h, ia = cam.setup_cam(cti=ctis[0], fps=FPS)

# Start the acquisition engine:

width = ia.remote_device.node_map.Width.value
height = ia.remote_device.node_map.Height.value
frames = np.zeros([T*FPS, height, width], dtype=np.uint8)

# warmup
for i in tqdm.tqdm(range(10), desc="warmup"):
    with ia.fetch() as buffer:
        pass

started = datetime.datetime.now()
try:
    for i in tqdm.tqdm(range(T*FPS), desc="Fetching frames"):
        with ia.fetch() as buffer:
            np.copyto(frames[i], buffer.payload.components[0].data.reshape(buffer.payload.components[0].height, buffer.payload.components[0].width))
except Exception as e:
    print(e)
finally:
    cam.disconnect_cam(h, ia)

timestamp = started.strftime("%Y%m%d%H%M%S")
filename = f"{timestamp}_{FPS}fps.h5"
with h5py.File(filename, "w") as f:
    f.create_dataset("frames", data=frames, compression="gzip", compression_opts=9)
    f.attrs["recording_id"] = str(uuid.uuid4())
    f.attrs["timestamp"] = timestamp  # start of the recording
