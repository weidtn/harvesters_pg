#!/usr/bin/env python3
"""Diagnose camera AOI changes on a frame grabber.

THIS SCRIPT CHANGES THE CAMERA: it opens the first GenICam camera, streams a
few frames, shrinks its Width/Height a little, streams again, and restores
the original AOI. Close harvesters_pg (or any other program using the camera)
first.

Every AOI change takes the safe order only (cam.change_aoi, as in the GUI):
quiesced stop (camera AcquisitionStop, drain until no frame arrives, then
the stream stop) -> write the AOI to the stopped camera -> destroy the
ImageAcquirer -> create a new one, at whose open the frame grabber latches
the camera's new size -> check camera == grabber. Before every start the
check is printed (`start check: ... -> OK/REFUSED`); when it is refused,
nothing is started and the probe ends. On exit the original AOI is restored
the same way, so camera and grabber agree again.

--try-grabber-write (off by default; UNVERIFIED on the crash-prone rig)
adds one step that writes the grabber's own ROI nodes (Width/Height,
PHX_ROI_XLENGTH/PHX_ROI_YLENGTH) on the stopped, still open camera instead
of re-opening it. That step starts only if the check passes afterwards.

At every step it prints what the camera (Width, Height, OffsetX, OffsetY,
PayloadSize), the frame grabber (GenTL device module) and the data streams
report. The camera keeps its other settings.

Run:  uv run aoi_probe.py [--try-grabber-write]
"""

import argparse
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


def start(ia, label: str) -> bool:
    """Start after the camera == grabber check; False (nothing started) if refused."""
    print(f"--- start ({label})")
    try:
        cam.start_acquisition(ia)  # logs the `start check: ...` line
    except cam.StartRefused as e:
        print(f"  NOT STARTED: {e}")
        return False
    show(ia, f"after start ({label})")
    return True


def change(h, ia, key, aoi):
    """The safe AOI change, see cam.change_aoi. Returns the new ImageAcquirer.

    *ia* is unusable afterwards, also when this raises (both are released).
    """
    print(f"=== AOI change to {aoi} (width, height, offset_x, offset_y)")
    # The GUI's AOI cooldown: no back-to-back re-opens on the frame grabber.
    time.sleep(COOLDOWN_S)
    print("  quiesced stop -> write the AOI to the stopped camera -> destroy -> create -> check")
    previous = ia
    ia, applied = cam.change_aoi(h, ia, aoi, key)
    print(f"  applied {applied}")
    show(ia, "re-opened, before start" if ia is not previous else "AOI unchanged, not re-opened")
    return ia


def try_grabber_write(ia, aoi) -> bool:
    """Opt-in: write the camera AOI and the grabber's ROI nodes without re-opening.

    Returns True if the check passed and every frame arrived.
    """
    print(f"=== --try-grabber-write: AOI {aoi} WITHOUT a re-open (unverified)")
    time.sleep(COOLDOWN_S)
    cam.quiesce_and_stop(ia, destroy=False)
    print(f"  camera AOI written: {cam.set_resolution(ia, *aoi)}")
    with cam.IA_LOCK:
        grabber = ia.device.node_map
        for names in (("Width", "Height"), ("PHX_ROI_XLENGTH", "PHX_ROI_YLENGTH")):
            for name, value in zip(names, aoi[:2]):
                try:
                    node = getattr(grabber, name)
                    before = node.value
                    node.value = value
                    print(f"  grabber {name}: {before} -> {node.value}")
                except Exception as e:
                    print(f"  grabber {name}: not written ({type(e).__name__}: {e})")
    show(ia, "after the grabber write, before start")
    if not start(ia, "grabber write"):
        return False
    return fetch_frames(ia, f"grabber write {aoi[0]}x{aoi[1]}")


def smaller(value: int, axis: cam.AoiAxis) -> int:
    """*value* about 1/8 smaller, at least one increment, on the node's grid."""
    step = max(axis.size_inc, value // 8)
    return cam.snap_value(value - step, axis.size_min, value, axis.size_inc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--try-grabber-write",
        action="store_true",
        help="also try writing the grabber ROI nodes directly (unverified, off by default)",
    )
    args = parser.parse_args(argv)
    print(__doc__.split("\n\n")[1])
    if args.try_grabber_write:
        print("--try-grabber-write is ON: one step writes the frame grabber's ROI nodes directly.")
    print("Starting in 3 s - Ctrl+C to abort.")
    time.sleep(3)
    # INFO: the quiesce steps and every `start check: ...` line.
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s", stream=sys.stdout)

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
    key = {"display_name": name}
    print(f"Using the first camera: {name!r}")

    # Opened as it is: no connect defaults, so no re-open either.
    _, ia = cam.setup_cam(cti=ctis, search_key=key, start=False, configure=False)
    original = None
    ok = True
    grabber_written = False
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
        if not start(ia, "original AOI"):
            ok = False
            return 1
        ok &= fetch_frames(ia, "original AOI")

        target = (smaller(width, limits.x), smaller(height, limits.y), None, None)
        ia, previous = None, ia  # change() releases it, also on failure
        ia = change(h, previous, key, target)
        if not start(ia, f"new AOI {target[0]}x{target[1]}"):
            ok = False
            return 1
        ok &= fetch_frames(ia, f"new AOI {target[0]}x{target[1]}")
        show(ia, "after fetching at the new AOI")

        if args.try_grabber_write:
            grabber_written = True
            ok &= try_grabber_write(ia, original)
    except KeyboardInterrupt:
        print("Interrupted.")
        ok = False
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        ok = False
    finally:
        ok &= restore(h, ia, ctis, key, original, grabber_written)
        cam.release_harvester()
        print("RESULT:", "all frames arrived" if ok else "FAILED or SOME FRAMES MISSING - see above")
    return 0 if ok else 1


def restore(h, ia, ctis, key, original, force_reopen: bool) -> bool:
    """Leave camera and grabber consistent at *original*; True if verified.

    Re-opens after the restore write (cam.change_aoi), unless the camera is
    already at *original* and the grabber agrees - fewer open/close cycles.
    After --try-grabber-write (*force_reopen*) it always re-opens once, so
    the grabber is set up from scratch again.
    """
    if original is None:
        cam.quiesce_and_stop(ia)
        return True
    print("=== restoring the original AOI")
    try:
        if ia is None:  # an AOI change failed and released everything
            _, ia = cam.setup_cam(cti=ctis, search_key=key, start=False, configure=False)
        current = cam.current_aoi(ia)
        at_original = all(want is None or got == want for want, got in zip(original, current))
        if at_original and not force_reopen and cam.check_start(ia)[0]:
            print("  camera already at the original AOI and the grabber agrees; no re-open")
            return True
        ia, previous = None, ia
        if at_original and force_reopen:
            print("  camera at the original AOI; re-opening once to reset the grabber")
            time.sleep(COOLDOWN_S)
            cam.quiesce_and_stop(previous)
            _, ia = cam.setup_cam(cti=ctis, search_key=key, start=False, configure=False)
            show(ia, "re-opened, before start")
        else:
            ia = change(h, previous, key, original)
        if not start(ia, "restored AOI"):
            return False
        return fetch_frames(ia, "restored AOI")
    except Exception as e:
        print(f"Restoring the original AOI failed: {type(e).__name__}: {e}")
        return False
    finally:
        cam.quiesce_and_stop(ia)


if __name__ == "__main__":
    sys.exit(main())
