#!/usr/bin/env python3
"""Diagnose camera AOI changes on a frame grabber.

THIS SCRIPT CHANGES THE CAMERA: it opens the first GenICam camera, streams a
few frames, shrinks its Width/Height a little, streams again, and restores
the original AOI. Close harvesters_pg (or any other program using the camera)
first.

Every AOI change takes the safe path only: stop -> destroy the
ImageAcquirer (closes the data stream, revokes the buffers) -> create a new
one for the same camera -> write the AOI -> start. Stop -> write -> start on
one ImageAcquirer, which preceded fatal PCIe machine checks on the CoaXPress
rig, is never done.

At every step it prints what the camera (Width, Height, OffsetX, OffsetY,
PayloadSize), the frame grabber (GenTL device module) and the data streams
report, so a grabber that does not follow the camera's AOI shows up.
Nothing on the grabber is written; the camera keeps its other settings.

Run:  uv run aoi_probe.py
"""

import logging
import sys
import time

import cam

FETCH_TIMEOUT_S = 2.0
FRAMES = 3
COOLDOWN_S = 1.0  # pause before every re-open, like the GUI's AOI cooldown


def show(ia, label: str) -> None:
    """Print the AOI-related values of the camera, grabber and data streams."""
    print(f"--- {label}")
    for part, values in cam.aoi_snapshot(ia).items():
        print(f"  {part}:" + ("" if values else " (no matching nodes)"))
        for name, value in values.items():
            print(f"    {name} = {value}")


def fetch_frames(ia, label: str) -> bool:
    """Fetch FRAMES frames; print their shapes or the failure. True if all came."""
    print(f"--- fetching {FRAMES} frames ({label}), {FETCH_TIMEOUT_S:g} s timeout each")
    ok = True
    for i in range(FRAMES):
        started = time.monotonic()
        try:
            with cam.IA_LOCK:
                buffer = ia.try_fetch(timeout=FETCH_TIMEOUT_S)
                if buffer is None:
                    print(f"  frame {i}: none within {FETCH_TIMEOUT_S:g} s (or incomplete)")
                    ok = False
                    continue
                with buffer:
                    component = buffer.payload.components[0]
                    print(
                        f"  frame {i}: {component.width}x{component.height} "
                        f"{component.data_format}, {component.data.size} values, "
                        f"after {time.monotonic() - started:.3f} s"
                    )
        except Exception as e:
            print(f"  frame {i}: error {type(e).__name__}: {e}")
            ok = False
    return ok


def open_camera(ctis, name: str):
    """Open *name* without writing any connect defaults to it."""
    _, ia = cam.setup_cam(
        cti=ctis, search_key={"display_name": name}, start=False, configure=False
    )
    return ia


def reopen_with_aoi(ia, ctis, name: str, aoi):
    """The safe AOI change: stop -> destroy -> create -> write AOI -> start.

    Returns the new ImageAcquirer; *ia* is unusable afterwards.
    """
    print(f"=== AOI change to {aoi} (width, height, offset_x, offset_y)")
    # The GUI's AOI cooldown: no back-to-back re-opens on the frame grabber.
    time.sleep(COOLDOWN_S)
    print("  stop + destroy the image acquirer (closes the data stream, revokes buffers)")
    cam.teardown_cam(None, ia)
    print("  create a new image acquirer for the same camera")
    ia = open_camera(ctis, name)
    show(ia, "re-opened, before the AOI write")
    applied = cam.set_resolution(
        ia, width=aoi[0], height=aoi[1], offset_x=aoi[2], offset_y=aoi[3]
    )
    print(f"  applied {applied}")
    show(ia, "after the AOI write, before start")
    with cam.IA_LOCK:
        ia.start()
    show(ia, "after start")
    return ia


def smaller(value: int, axis: cam.AoiAxis) -> int:
    """*value* about 1/8 smaller, at least one increment, on the node's grid."""
    step = max(axis.size_inc, value // 8)
    return cam.snap_value(value - step, axis.size_min, value, axis.size_inc)


def main() -> int:
    print(__doc__.split("\n\n")[1])
    print("Starting in 3 s - Ctrl+C to abort.")
    time.sleep(3)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    ctis = cam.load_cti_envvar()
    if not ctis:
        print("No GenTL producer (*.cti) found; set GENICAM_GENTL64_PATH.")
        return 2
    print(f"GenTL producers: {ctis}")
    h = cam.get_harvester(ctis)
    h.update()
    names = [device.display_name for device in h.device_info_list]
    print(f"Devices: {names}")
    if not names:
        print("No camera found.")
        return 2
    name = names[0]
    print(f"Using the first camera: {name!r}")

    ia = open_camera(ctis, name)
    original = None
    ok = True
    try:
        nodes = ia.remote_device.node_map
        for context in ("DeviceModelName", "PixelFormat", "TriggerMode", "AcquisitionMode"):
            try:
                print(f"  {context} = {getattr(nodes, context).value}")
            except Exception as e:
                print(f"  {context}: not available ({type(e).__name__})")
        limits, (width, height, offset_x, offset_y) = cam.read_aoi(ia)
        original = (
            width,
            height,
            offset_x if limits.x.has_offset else None,
            offset_y if limits.y.has_offset else None,
        )
        print(f"  AOI limits: {limits}")
        print(f"  original AOI: {original}")
        show(ia, "connected, never started")
        with cam.IA_LOCK:
            ia.start()
        show(ia, "after start at the original AOI")
        ok &= fetch_frames(ia, "original AOI")

        target = (smaller(width, limits.x), smaller(height, limits.y), None, None)
        ia = reopen_with_aoi(ia, ctis, name, target)
        ok &= fetch_frames(ia, f"new AOI {target[0]}x{target[1]}")
        show(ia, "after fetching at the new AOI")
    except KeyboardInterrupt:
        print("Interrupted.")
        ok = False
    finally:
        if original is not None:
            try:
                print("=== restoring the original AOI")
                ia = reopen_with_aoi(ia, ctis, name, original)
                ok &= fetch_frames(ia, "restored AOI")
            except Exception as e:
                print(f"Restoring the original AOI failed: {type(e).__name__}: {e}")
                ok = False
        cam.teardown_cam(None, ia)
        cam.release_harvester()
    print("RESULT:", "all frames arrived" if ok else "SOME FRAMES MISSING - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
