"""Raw PHX (Active Silicon C SDK) camera acquisition backend.

Bypasses the GenTL/GenICam producer entirely for the acquisition hot path.

Why this exists
----------------
Investigation on this rig (see ``~/ASL-RMA-evidence-FBD-4XCXP12.md`` and
``~/ASL-driver-fix-summary.md`` in niwe's home directory on sweetums) found
that the FireBird FBD_4XCXP12_3PE8 frame grabber has a hardware defect that
produces a fatal, unrecoverable PCIe fault on its root port. It correlates
specifically with GenTL/GenCP control-plane traffic - register read/write
churn issued through the GenTL producer during configure/start/stop cycles -
and was never observed with sustained raw-PHX streaming, even at full CXP-12
line rate (the RMA doc's differential experiment: ``phx_rate``, Active
Silicon's own raw-PHX example, survived 1.46 million buffers with zero
errors; a harvesters-based GenCP-churn script died within minutes).

This module follows the same split the investigation's own ``mini_acq.c`` /
``phx_snap.c`` reproducers (``/usr/local/activesilicon/source/examples/c/``)
use:

1. A brief, ONE-SHOT GenICam session (via the existing, already-hardened
   :mod:`cam` module) configures the camera - exposure, gain, resolution,
   frame rate - and is then fully torn down (GenTL producer unloaded)
   *before* any acquisition happens. This is the only GenICam traffic this
   module ever issues, and it happens once per connect, never per frame and
   never while streaming.
2. All actual image acquisition goes through the raw PHX C API
   (``libphxapi-x86_64.so`` via ``ctypes``): ``PHX_Create``/``PHX_Open`` for
   the board, ``PHX_StreamRead`` for frame delivery, and a plain CXP
   bootstrap register write (``PHX_ControlWrite``) - discovered the same way
   Active Silicon's own ``PhxCommonUpdateCxpRegs()`` example helper does -
   to start/stop the camera's acquisition engine. No GenTL producer is
   loaded for the lifetime of the raw-PHX session.

IMPORTANT, measured 2026-09-21: this only protects sustained STREAMING, not
the connect step itself. configure_camera_via_genicam() still makes one
real GenTL round trip per connect, and a stress test that repeatedly
connects/disconnects (stress_phx.py, mirroring stress_ui.py against ui.py)
crashed the machine on cycle 1 - marginally FASTER than the equivalent
ui.py test, which survived 4 cycles first. Both implementations hit the
same underlying defect through connect/disconnect churn, because both run
that GenICam round trip on every connect regardless of which API streams
the frames afterward. The raw-PHX path's demonstrated, repeatable safety
margin is for staying connected and streaming/reconfiguring (60 frames at
a steady 30 fps, multiple live reconfigure cycles, all crash-free in the
same testing session) - not for rapid connect/disconnect cycling, which
remains dangerous on this hardware no matter which API is used.

Constants below were extracted by compiling a small C program against the
installed ``/usr/local/activesilicon/include/phx_api.h`` and printing the
resolved values, not hand-transcribed from the header's bit-macro
definitions - that avoids a silently-wrong magic number talking to real
hardware. See the dump program used to generate them:
``PHX_BOARD_NUMBER``, ``PHX_START`` etc. below match
``phx_api.h`` v08.78.02 exactly as built.
"""

from __future__ import annotations

import logging
import threading
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int32,
    c_uint8,
    c_uint32,
    c_void_p,
    cast,
)

import numpy as np

import cam  # reuse the already-hardened GenICam setup/teardown for one-shot configuration

_LIB_PATH = "/usr/local/activesilicon/lib64/libphxapi-x86_64.so"

# --- constants extracted from phx_api.h (see module docstring) -------------
PHX_BOARD_NUMBER = 0xC0070300
PHX_CHANNEL_NUMBER = 0xC0070400
PHX_CONFIG_MODE = 0xC0070500
PHX_ROI_XLENGTH = 0xC0025B00
PHX_ROI_YLENGTH = 0xC0025C00
PHX_BUS_FORMAT = 0xC0022600
PHX_ACQ_BLOCKING = 0xC0029300

PHX_BOARD_NUMBER_1 = 0xC0010001
PHX_CHANNEL_NUMBER_1 = 0xC0010010
PHX_CONFIG_NORMAL = 0xC0010000
PHX_ENABLE = 0x00000001
PHX_DISABLE = 0x00000002
PHX_BUS_FORMAT_MONO8 = 0xC0022601

PHX_START = 0xC0030100
PHX_ABORT = 0xC0030500
PHX_BUFFER_GET = 0xC0030600

PHX_REGISTER_DEVICE = 0xC0080002

PHX_INTRPT_BUFFER_READY = 0x00000002

PHX_OK = 0

# CXP bootstrap register addresses (fixed by the CoaXPress standard, not
# vendor/camera specific - the same ones PhxCommonUpdateCxpRegs() in the
# SDK's own example common.c reads).
_CXP_BOOTSTRAP_REVISION = 0x0004
_CXP_BOOTSTRAP_ACQ_START_PTR = 0x300C
_CXP_BOOTSTRAP_ACQ_STOP_PTR = 0x3010

_BUS_FORMAT_BY_PIXEL_FORMAT = {
    "Mono8": PHX_BUS_FORMAT_MONO8,
}

# Upper bound, in seconds, for every buffer fetch in this module - same
# rationale as cam.FETCH_TIMEOUT_S: a camera that never delivers must not
# hang the caller forever.
FETCH_TIMEOUT_S = 5.0


class PhxError(RuntimeError):
    """Raised when a PHX_* call returns a non-PHX_OK status."""


class stImageBuff(Structure):
    _fields_ = [("pvAddress", c_void_p), ("pvContext", c_void_p)]


_lib = CDLL(_LIB_PATH)

_etFctErrorHandler = CFUNCTYPE(None, c_char_p, c_int32, c_char_p)
_PhxCallback = CFUNCTYPE(None, c_void_p, c_uint32, c_void_p)

_lib.PHX_Create.argtypes = [POINTER(c_void_p), c_void_p]  # NULL error handler passed as plain void*; see open_board()
_lib.PHX_Create.restype = c_int32
_lib.PHX_Open.argtypes = [c_void_p]
_lib.PHX_Open.restype = c_int32
_lib.PHX_Close.argtypes = [POINTER(c_void_p)]
_lib.PHX_Close.restype = c_int32
_lib.PHX_Destroy.argtypes = [POINTER(c_void_p)]
_lib.PHX_Destroy.restype = c_int32
_lib.PHX_ParameterSet.argtypes = [c_void_p, c_int32, c_void_p]
_lib.PHX_ParameterSet.restype = c_int32
_lib.PHX_ParameterGet.argtypes = [c_void_p, c_int32, c_void_p]
_lib.PHX_ParameterGet.restype = c_int32
_lib.PHX_StreamRead.argtypes = [c_void_p, c_int32, c_void_p]
_lib.PHX_StreamRead.restype = c_int32
_lib.PHX_ControlRead.argtypes = [c_void_p, c_int32, c_void_p, POINTER(c_uint8), POINTER(c_uint32), c_uint32]
_lib.PHX_ControlRead.restype = c_int32
_lib.PHX_ControlWrite.argtypes = [c_void_p, c_int32, c_void_p, POINTER(c_uint8), POINTER(c_uint32), c_uint32]
_lib.PHX_ControlWrite.restype = c_int32


def _check(stat: int, what: str) -> None:
    if stat != PHX_OK:
        raise PhxError(f"{what} failed with status 0x{stat & 0xFFFFFFFF:08X}")


def _param_set_i32(handle: c_void_p, param: int, value: int) -> None:
    v = c_int32(value)
    _check(_lib.PHX_ParameterSet(handle, param, byref(v)), f"PHX_ParameterSet(0x{param:08X})")


def _read_cxp_reg(handle: c_void_p, address: int, timeout_ms: int = 800) -> int:
    """Read one big-endian 32 bit CXP bootstrap/device register."""
    addr = c_uint32(address)
    data = (c_uint8 * 4)()
    size = c_uint32(4)
    _check(
        _lib.PHX_ControlRead(handle, PHX_REGISTER_DEVICE, byref(addr), data, byref(size), timeout_ms),
        f"PHX_ControlRead(0x{address:08X})",
    )
    return int.from_bytes(bytes(data), byteorder="big")


def _write_cxp_reg(handle: c_void_p, address: int, value: int, timeout_ms: int = 800) -> None:
    """Write one big-endian 32 bit CXP bootstrap/device register."""
    addr = c_uint32(address)
    data = (c_uint8 * 4)(*value.to_bytes(4, byteorder="big"))
    size = c_uint32(4)
    _check(
        _lib.PHX_ControlWrite(handle, PHX_REGISTER_DEVICE, byref(addr), data, byref(size), timeout_ms),
        f"PHX_ControlWrite(0x{address:08X})",
    )


def _discover_acq_start_stop(handle: c_void_p) -> tuple[int, int]:
    """Return (AcquisitionStart address, AcquisitionStop address).

    Same algorithm as Active Silicon's own PhxCommonUpdateCxpRegs() example
    helper: CXP >= v1.1 cameras publish the address of their own Start/Stop
    command registers at these two fixed bootstrap pointers.
    """
    revision = _read_cxp_reg(handle, _CXP_BOOTSTRAP_REVISION)
    if revision < 0x00010001:
        raise PhxError(
            f"Camera reports CXP revision 0x{revision:08X} (< v1.1); "
            "AcquisitionStart/Stop bootstrap pointers are not available."
        )
    start_addr = _read_cxp_reg(handle, _CXP_BOOTSTRAP_ACQ_START_PTR)
    stop_addr = _read_cxp_reg(handle, _CXP_BOOTSTRAP_ACQ_STOP_PTR)
    return start_addr, stop_addr


def configure_camera_via_genicam(
    cti: str,
    fps: int = 30,
    exposure: int = 999,
    blacklevel: int = 125,
    gain: float | None = None,
    width: int | None = None,
    height: int | None = None,
    search_key: int | dict | None = None,
) -> tuple[int, int, str]:
    """Configure the camera once via GenICam, then fully release the GenTL
    producer before returning.

    This is the ONLY GenICam/GenTL traffic this module issues: one connect,
    one set of writes (delegated to the already-hardened
    ``cam.setup_cam``/``cam._configure_cam``), one clean teardown that
    unloads the producer. Returns the (width, height, pixel_format) the
    camera actually ended up at, so the raw-PHX side can be opened with a
    matching ROI/format.
    """
    h, ia = cam.setup_cam(
        cti,
        height=height,
        width=width,
        fps=fps,
        blacklevel=blacklevel,
        exposure=exposure,
        search_key=search_key,
    )
    try:
        nodes = ia.remote_device.node_map
        if gain is not None:
            try:
                nodes.Gain.value = gain
            except Exception as e:
                logging.warning(f"Could not set Gain to {gain}: {e}")
        actual_width = int(nodes.Width.value)
        actual_height = int(nodes.Height.value)
        pixel_format = str(nodes.PixelFormat.value)
    finally:
        # Full teardown, including the shared harvester: the raw PHX side
        # needs the board completely free, not just "disconnected"
        # (cam.disconnect_cam deliberately leaves the producer loaded for a
        # fast reconnect - the wrong tradeoff here, since nothing may keep
        # touching the producer while raw PHX is streaming).
        cam.teardown_cam(h, ia)
    return actual_width, actual_height, pixel_format


class PhxAcquirer:
    """One open PHX board/channel plus its CXP acquisition control.

    Mirrors just enough of harvesters' ``ImageAcquirer`` surface
    (``try_fetch()``, ``is_acquiring()``, ``stop()``, ``destroy()``) for
    VideoThread-style callers - but every call here goes straight to the PHX
    C API. No GenTL producer is loaded for the lifetime of this object.
    """

    def __init__(self, handle: c_void_p, width: int, height: int, acq_start_addr: int, acq_stop_addr: int) -> None:
        self._handle = handle
        self.width = width
        self.height = height
        self._acq_start_addr = acq_start_addr
        self._acq_stop_addr = acq_stop_addr
        self._buffer_ready = threading.Event()
        # Kept alive on self: ctypes does not keep a reference to a CFUNCTYPE
        # callback on its own, and a garbage-collected trampoline crashes the
        # process the next time the driver calls it.
        self._callback = _PhxCallback(self._on_interrupt)
        self._streaming = False
        self._camera_acquiring = False

    def _on_interrupt(self, _handle: int, mask: int, _params: int) -> None:
        if mask & PHX_INTRPT_BUFFER_READY:
            self._buffer_ready.set()

    def start(self) -> None:
        """Start frame-grabber streaming, then start the camera itself."""
        if self._streaming:
            return
        self._buffer_ready.clear()
        _check(_lib.PHX_StreamRead(self._handle, PHX_START, self._callback), "PHX_StreamRead(START)")
        self._streaming = True
        _write_cxp_reg(self._handle, self._acq_start_addr, 1)
        self._camera_acquiring = True

    def stop(self) -> None:
        """Stop the camera, then abort frame-grabber streaming.

        Each half is guarded on its own: a raising register write must not
        skip the streaming abort, or the frame grabber is left mid-DMA with
        no consumer, which is exactly the kind of dangling state this whole
        module exists to avoid.
        """
        if self._camera_acquiring:
            try:
                _write_cxp_reg(self._handle, self._acq_stop_addr, 1)
            except PhxError as e:
                logging.warning(f"Could not write AcquisitionStop register: {e}")
            self._camera_acquiring = False
        if self._streaming:
            try:
                _check(_lib.PHX_StreamRead(self._handle, PHX_ABORT, None), "PHX_StreamRead(ABORT)")
            except PhxError as e:
                logging.warning(f"PHX_StreamRead(ABORT): {e}")
            self._streaming = False

    def is_acquiring(self) -> bool:
        return self._streaming

    def try_fetch(self, timeout: float = FETCH_TIMEOUT_S) -> np.ndarray | None:
        """Return the latest frame as an owned uint8 (height, width) array,
        or None on timeout - matching ``cam.get_harvester_image()``'s
        contract so callers need no changes to their timeout handling.
        """
        if not self._buffer_ready.wait(timeout):
            return None
        self._buffer_ready.clear()
        buf = stImageBuff()
        stat = _lib.PHX_StreamRead(self._handle, PHX_BUFFER_GET, byref(buf))
        if stat != PHX_OK or not buf.pvAddress:
            return None
        src = cast(buf.pvAddress, POINTER(c_uint8 * (self.width * self.height))).contents
        # np.ctypeslib.as_array() would return a view straight into the
        # frame grabber's DMA ring buffer, which hardware reuses as soon as
        # the next buffer-ready interrupt fires (ACQ_BLOCKING is disabled
        # precisely so it can). Copy now, while it is still this frame's.
        return np.frombuffer(src, dtype=np.uint8).reshape(self.height, self.width).copy()

    def destroy(self) -> None:
        """Release the board: stop acquisition, close, destroy the handle."""
        self.stop()
        if self._handle is None:
            return
        try:
            _check(_lib.PHX_Close(byref(self._handle)), "PHX_Close")
        except PhxError as e:
            logging.warning(f"PHX_Close: {e}")
        try:
            _check(_lib.PHX_Destroy(byref(self._handle)), "PHX_Destroy")
        except PhxError as e:
            logging.warning(f"PHX_Destroy: {e}")
        self._handle = None


def open_board(
    width: int,
    height: int,
    pixel_format: str = "Mono8",
    board_number: int = PHX_BOARD_NUMBER_1,
    channel_number: int = PHX_CHANNEL_NUMBER_1,
) -> PhxAcquirer:
    """Open the frame grabber via the raw PHX API for *width* x *height*
    frames in *pixel_format*.

    The camera itself must already be configured and left at this exact
    resolution/format - see :func:`configure_camera_via_genicam`. PHX only
    configures the frame grabber's DMA/ROI side; the only camera register
    this module ever touches directly is AcquisitionStart/Stop (see
    ``PhxAcquirer.start()``/``stop()``).
    """
    bus_format = _BUS_FORMAT_BY_PIXEL_FORMAT.get(pixel_format)
    if bus_format is None:
        raise PhxError(f"Unsupported pixel format for the raw PHX path: {pixel_format!r}")

    h = c_void_p()
    _check(_lib.PHX_Create(byref(h), None), "PHX_Create")
    try:
        _param_set_i32(h, PHX_BOARD_NUMBER, board_number)
        _param_set_i32(h, PHX_CONFIG_MODE, PHX_CONFIG_NORMAL)
        _param_set_i32(h, PHX_CHANNEL_NUMBER, channel_number)
        _check(_lib.PHX_Open(h), "PHX_Open")
        _param_set_i32(h, PHX_ROI_XLENGTH, width)
        _param_set_i32(h, PHX_ROI_YLENGTH, height)
        _param_set_i32(h, PHX_BUS_FORMAT, bus_format)
        # Let the frame grabber reuse DMA buffers freely instead of waiting
        # for software to mark them free - same setting mini_acq.c and
        # phx_rate.c use, and PhxAcquirer.try_fetch() copies out before the
        # next interrupt can overwrite a buffer.
        _param_set_i32(h, PHX_ACQ_BLOCKING, PHX_DISABLE)
        acq_start_addr, acq_stop_addr = _discover_acq_start_stop(h)
        logging.info(
            f"Raw PHX: board opened, {width}x{height} {pixel_format}, "
            f"AcquisitionStart=0x{acq_start_addr:08X} AcquisitionStop=0x{acq_stop_addr:08X}"
        )
    except Exception:
        _lib.PHX_Close(byref(h))
        _lib.PHX_Destroy(byref(h))
        raise
    return PhxAcquirer(h, width, height, acq_start_addr, acq_stop_addr)


def get_image(pa: PhxAcquirer) -> np.ndarray:
    """Get one frame, or a black frame on timeout (never raises)."""
    frame = pa.try_fetch(timeout=FETCH_TIMEOUT_S)
    if frame is None:
        logging.warning(f"No frame within {FETCH_TIMEOUT_S}s; showing a black image")
        return np.zeros((pa.height, pa.width), dtype=np.uint8)
    return frame


def record_frames(pa: PhxAcquirer, n_frames: int, progress=None, preview=None) -> np.ndarray:
    """Record n_frames from the frame grabber into a preallocated array.

    Starts acquisition unconditionally and stops it in a finally block,
    matching cam.record_harvester_frames()'s contract: callers (e.g. the
    GUI's on_btn_acquire_clicked, which pauses live view first) must not
    have to know or care whether the board happens to already be
    streaming.
    """
    if progress is not None:
        progress.setMaximum(1 if n_frames == 1 else n_frames - 1)
    frames = np.empty((n_frames, pa.height, pa.width), dtype=np.uint8)
    pa.start()
    try:
        for i in range(n_frames):
            frame = pa.try_fetch(timeout=FETCH_TIMEOUT_S)
            if frame is None:
                raise TimeoutError(f"The camera delivered no frame within {FETCH_TIMEOUT_S}s while recording.")
            frames[i] = frame
            if preview is not None:
                preview(frames[i])
            if progress is not None:
                progress.setValue(1 if n_frames == 1 else i)
    finally:
        # Stop on every exit, including a fetch timeout: a board left
        # acquiring keeps filling buffers nobody drains.
        pa.stop()
    return frames
