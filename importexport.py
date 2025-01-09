#!/usr/bin/env python3
import numpy as np
import h5py
import cv2


def save_frames_to_h5(
        frames: np.ndarray, filename: str, complevel: int = 4, metadata = {}
) -> None:
    """Save frames to an h5 file.

    Args:
        frames (np.ndarray): The frames to save
        filename (str): The name of the file to save the frames to
        complevel (int): The compression level to use (4 is default, 0 is no compression)
        metadata (dict): The metadata to save with the frames
    """
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset(
            "frames",
            data=frames.transpose([0, 2, 1]),
            compression="gzip",
            compression_opts=complevel,
        )
        [dset.attrs.create(k, v) for k, v in metadata.items()]


def save_frames_to_png(frames: np.ndarray, filename: str) -> None:
    """Save frames to a png file.

    Args:
        frames (np.ndarray): The frames to save
        filename (str): The name of the file to save the frames to
    """
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{filename}_{i}.png", frame.T)


def load_frames_from_h5(filename) -> np.ndarray:
    """Load frames from an h5 file.

    Args:
        filename (str): The name of the file to load the frames from

    Returns:
        frames: The loaded frames
    """
    with h5py.File(filename, "r") as f:
        frames = f["frames"][:].transpose([0, 2, 1])
    return frames
