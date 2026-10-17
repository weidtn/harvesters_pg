#!/usr/bin/env python3
import numpy as np
import h5py
import cv2
from PyQt6.QtWidgets import QProgressBar
import os
import pathlib
import re

import cam


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
    # Work on a copy: popping the datasets out of the caller's dict removed
    # them from it, so a second save of the same recording lost them.
    metadata = dict(metadata)
    try:
        with h5py.File(filename, "w") as f:
            f.create_dataset(
                "frames",
                data=frames,
                compression="gzip",
                compression_opts=complevel,
            )
            for name in ("timestamps", "frame_counters"):
                if name in metadata:
                    f.create_dataset(
                        name,
                        data=metadata.pop(name),
                        compression="gzip",
                        compression_opts=complevel,
                    )
            for k, v in metadata.items():
                try:
                    f.attrs.create(k, v)
                except Exception:
                    pass
    finally:
        # Also on a failed write: the bar used to stay busy with "Saving...".
        if progressbar is not None:
            progressbar.setMaximum(1)
            progressbar.setValue(1)
            progressbar.setFormat("%p%")


def create_path(filename: str) -> None:
    """Create a path for the given filename. The file ending is removed.

    Args:
        path (str): The path to create
    """
    # Get the path without the file ending

    path = pathlib.Path(filename)
    new_path = f"{path.parent}/{path.name.split('.')[0]}"
    pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)
    return new_path


def _write_image(path: str, frame: np.ndarray) -> None:
    if not cv2.imwrite(path, frame):
        raise OSError(f"Could not write {path}")


def _remove_stale_frames(prefix: str, extension: str, n_frames: int) -> None:
    """Delete numbered frames an earlier, longer export left in the folder.

    Only files named exactly like this export's frames (<prefix>_<n>.<ext>)
    with n beyond the new frame count are removed; otherwise the folder mixed
    the new recording with the tail of an old one.
    """
    pattern = re.compile(re.escape(os.path.basename(prefix)) + r"_(\d+)\." + extension + "$")
    folder = os.path.dirname(prefix) or "."
    for name in os.listdir(folder):
        match = pattern.match(name)
        if match and int(match.group(1)) >= n_frames:
            os.remove(os.path.join(folder, name))


def _save_frames_as_images(
    frames: np.ndarray, filename: str, extension: str, progressbar: QProgressBar | None
) -> None:
    suffix = f".{extension}"
    if suffix not in filename:
        filename += suffix
    path = create_path(filename)
    new_filename = f"{path}/{filename.split('/')[-1]}"
    # Save a single frame as a single image:
    if frames.ndim == 2:
        _write_image(new_filename, frames)
        return
    if progressbar is not None:
        if len(frames) == 1:
            progressbar.setMaximum(2)
        else:
            progressbar.setMaximum(len(frames))
    prefix = new_filename.split(suffix)[0]
    for i, frame in enumerate(frames):
        _write_image(f"{prefix}_{i}{suffix}", frame)
        if progressbar is not None:
            if len(frames) == 1:
                progressbar.setValue(2)
            else:
                progressbar.setValue(i)
    _remove_stale_frames(prefix, extension, len(frames))


def save_frames_to_png(
    frames: np.ndarray, filename: str, progressbar: QProgressBar | None = None
) -> None:
    """Save frames to png files, one per frame, in a folder named after *filename*.

    Args:
        frames (np.ndarray): The frames to save
        filename (str): The name of the file to save the frames to
        progressbar (QProgressBar): The progress bar to update

    Raises:
        OSError: if an image could not be written.
    """
    _save_frames_as_images(frames, filename, "png", progressbar)


def save_frames_to_jpg(
    frames: np.ndarray, filename: str, progressbar: QProgressBar | None = None
) -> None:
    """Save frames to jpg files, one per frame, in a folder named after *filename*.

    Args:
        frames (np.ndarray): The frames to save
        filename (str): The name of the file to save the frames to
        progressbar (QProgressBar): The progress bar to update

    Raises:
        OSError: if an image could not be written.
    """
    _save_frames_as_images(frames, filename, "jpg", progressbar)


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
    try:
        with h5py.File(filename, "r") as f:
            dataset = f["frames"]
            # Refuse before reading: a file larger than the free RAM would push
            # the machine into swap while it loads.
            cam.ensure_memory_for(dataset.size * dataset.dtype.itemsize, "file")
            frames = dataset[:]
            for k, v in f.attrs.items():  # written by save_frames_to_h5
                metadata[k] = v
            for k, v in dataset.attrs.items():
                metadata[k] = v
            if "timestamps" in f:
                metadata["timestamps"] = f["timestamps"][:]
            if "frame_counters" in f:
                metadata["frame_counters"] = f["frame_counters"][:]
    finally:
        if progressbar is not None:
            progressbar.setMaximum(1)
            progressbar.setValue(1)
            progressbar.setFormat("%p%")
    return frames, metadata
