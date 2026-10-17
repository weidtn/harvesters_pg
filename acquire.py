#!/usr/bin/env python3
import datetime
import uuid

import h5py

import cam

FPS = 1000
T = 1
PARTICLES = ""  # lab ids of the particle batches, comma-separated (optional)
OBJECTIVE = ""  # lab id of the objective (optional)


ctis = cam.load_cti_envvar()
h, ia = cam.setup_cam(cti=ctis, fps=FPS, start=False)
try:
    started = datetime.datetime.now()
    try:
        frames = cam.record_harvester_frames(ia, T * FPS, None)
    except cam.RecordingInterrupted as e:
        frames = e.frames
        print(f"Warning: {e}; keeping the {len(frames)} frames recorded before the camera stopped")
finally:
    cam.teardown_cam(h, ia)

timestamp = started.strftime("%Y%m%d%H%M%S")
filename = f"{timestamp}_{FPS}fps.h5"
with h5py.File(filename, "w") as f:
    f.create_dataset("frames", data=frames, compression="gzip", compression_opts=9)
    f.attrs["recording_id"] = str(uuid.uuid4())
    f.attrs["timestamp"] = timestamp  # start of the recording
    if PARTICLES:
        f.attrs["particles"] = PARTICLES
    if OBJECTIVE:
        f.attrs["objective"] = OBJECTIVE
