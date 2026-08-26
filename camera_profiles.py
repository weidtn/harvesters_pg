"""Per-camera GenICam quirk profiles.

Not every camera speaks the same GenICam feature names: SFNC-conformant
cameras use ExposureTime / AcquisitionFrameRate, while classic Allied Vision
firmware (e.g. Mako G, EoSenS) uses the legacy *Abs names instead
(ExposureTimeAbs, AcquisitionFrameRateAbs). A profile maps the abstract
features the rest of the code talks about onto the concrete node names of one
camera family.

Adding a camera family: append a CameraProfile entry to PROFILES whose
exposure and frame-rate nodes are named as that camera spells them.
detect_profile() picks the first entry whose nodes exist on the connected
camera, falling back to the SFNC default.
"""

from dataclasses import dataclass
import logging

# SFNC auto-mode enumeration values; identical across vendors.
AUTO_OFF = "Off"
AUTO_ON = "Continuous"


@dataclass(frozen=True)
class CameraProfile:
    """Node-name mapping for one camera family."""

    key: str
    exposure_node: str = "ExposureTime"
    frame_rate_node: str = "AcquisitionFrameRate"
    # Cameras without such a node (Allied Vision classic firmware) carry the
    # limit on the frame-rate node itself; see max_frame_rate().
    frame_rate_max_node: str | None = "AcquisitionFrameRateMax"
    gain_node: str = "Gain"
    # None = this family has no such auto feature (checkbox stays hidden).
    auto_exposure_node: str | None = "ExposureAuto"
    auto_gain_node: str | None = "GainAuto"


SFNC_DEFAULT = CameraProfile(key="sfnc")

# One entry covers all classic-firmware Allied Vision GigE cameras tested so
# far (Mako G-125B; EoSenS expected to match the same *Abs naming).
ALLIED_VISION_CLASSIC = CameraProfile(
    key="allied-vision-classic",
    exposure_node="ExposureTimeAbs",
    frame_rate_node="AcquisitionFrameRateAbs",
    frame_rate_max_node=None,
)

PROFILES = (ALLIED_VISION_CLASSIC, SFNC_DEFAULT)


def _has(nodemap, name: str | None) -> bool:
    if not name:
        return False
    try:
        getattr(nodemap, name)
        return True
    except Exception:
        return False


def detect_profile(nodemap) -> CameraProfile:
    """Return the profile whose nodes exist on the connected camera."""
    for profile in PROFILES:
        if _has(nodemap, profile.exposure_node) and _has(
            nodemap, profile.frame_rate_node
        ):
            return profile
    return SFNC_DEFAULT


def max_frame_rate(nodemap, profile: CameraProfile) -> float | None:
    """Return the camera's maximum frame rate, or None if it reports none.

    Prefers the dedicated max node (SFNC cameras). Allied Vision classic
    firmware has none, but carries the limit on the frame-rate node itself.
    """
    if profile.frame_rate_max_node:
        try:
            return float(getattr(nodemap, profile.frame_rate_max_node).value)
        except Exception as e:
            logging.debug(f"{profile.frame_rate_max_node} not available: {e}")
    try:
        return float(getattr(nodemap, profile.frame_rate_node).max)
    except Exception as e:
        logging.debug(f"{profile.frame_rate_node}.max not available: {e}")
        return None
