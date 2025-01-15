#!/usr/bin/env python3
import numpy as np
import h5py
import cv2
from PyQt6.QtWidgets import QProgressBar


def save_frames_to_h5(
    frames: np.ndarray,
    filename: str,
    complevel: int = 4,
    metadata={},
    progressbar: QProgressBar | None = None,
) -> None:
    """Save frames to an h5 file.

    Args:
        frames (np.ndarray): The frames to save
        filename (str): The name of the file to save the frames to
        complevel (int): The compression level to use (4 is default, 0 is no compression)
        metadata (dict): The metadata to save with the frames
        progressbar (QProgressBar): The progress bar to update
    """
    if progressbar is not None:
        progressbar.setFormat("Saving...")
        progressbar.setMaximum(0)
        progressbar.setValue(0)
        progressbar.setTextVisible(True)
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset(
            "frames",
            data=frames.transpose([0, 2, 1]),
            compression="gzip",
            compression_opts=complevel,
        )
        if "timestamps" in metadata:
            f.create_dataset(
                "timestamps",
                data=metadata["timestamps"],
                compression="gzip",
                compression_opts=complevel,
            )
            metadata.pop("timestamps")
        if "frame_counters" in metadata:
            f.create_dataset(
                "frame_counters",
                data=metadata["frame_counters"],
                compression="gzip",
                compression_opts=complevel,
            )
            metadata.pop("frame_counters")
        for k, v in metadata.items():
            try:
                dset.attrs.create(k, v)
            except Exception as e:
                pass
    if progressbar is not None:
        progressbar.setMaximum(1)
        progressbar.setValue(1)
        progressbar.setFormat("%p%")


def save_frames_to_png(
    frames: np.ndarray, filename: str, progressbar: QProgressBar | None = None
) -> None:
    """Save frames to a png file.

    Args:
        frames (np.ndarray): The frames to save
        filename (str): The name of the file to save the frames to
        progressbar (QProgressBar): The progress bar to update
    """
    # Save a single frame as a single image:
    if ".png" not in filename:
        filename += ".png"
    if frames.ndim == 2:
        cv2.imwrite(f"{filename}", frames.T)
        return
    if progressbar is not None:
        progressbar.setMaximum(len(frames) - 1)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{filename.split('.png')[0]}_{i}.png", frame.T)
        if progressbar is not None:
            progressbar.setValue(i)


def load_frames_from_h5(
    filename: str, progressbar: QProgressBar | None = None
) -> tuple[np.ndarray, dict]:
    """Load frames from an h5 file.

    Args:
        filename (str): The name of the file to load the frames from
        progressbar (QProgressBar): The progress bar to update


    Returns:
        tuple[np.ndarray, dict]: The frames and the metadata
    """
    metadata = {}
    if progressbar is not None:
        progressbar.setFormat("Loading...")
        progressbar.setMaximum(0)
        progressbar.setValue(0)
    with h5py.File(filename, "r") as f:
        frames = f["frames"][:].transpose([0, 2, 1])
        for k, v in f["frames"].attrs.items():
            metadata[k] = v
        if "timestamps" in f:
            metadata["timestamps"] = f["timestamps"][:]
        if "frame_counters" in f:
            metadata["frame_counters"] = f["frame_counters"][:]
    if progressbar is not None:
        progressbar.setMaximum(1)
        progressbar.setValue(1)
        progressbar.setFormat("%p%")
    return frames, metadata
