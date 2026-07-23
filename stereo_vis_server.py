#!/usr/bin/env python3

"""
Minimal stereo AprilTag server for the Vision Pi (monochrome cameras).

This is a corrected version of the previous server. Fixes applied:

FRAME / MATH FIXES
1.  Triangulated points are produced in the RECTIFIED left-camera frame
    (rotated by R1 relative to the physical left camera). They are now
    rotated back into the PHYSICAL left-camera frame with R1^T before any
    position or orientation math. This removes the constant fake yaw and
    the distance-dependent x error that the old constant offsets were
    fighting.
2.  The stereo-midpoint shift now uses the CALIBRATED baseline (from the
    calibration file / P2), not a hardcoded 50.0 mm, and the midpoint
    vector is rotated into the physical camera frame as well.
3.  Triangulation units are now explicit. Set CALIB_UNITS below to match
    the units of your chessboard square size during stereoCalibrate.
    A startup sanity check prints and validates the baseline.
4.  Orientation is now a proper, consistent decomposition:
    R = Ry(yaw) * Rx(pitch) * Rz(roll) in the camera frame
    (+X right, +Y down, +Z forward). Yaw/pitch keep the exact same
    convention as before (yaw = atan2(nx, nz), pitch = atan2(-ny, ...)),
    so the Control Pi sees compatible values, but roll is now extracted
    after removing yaw and pitch instead of being an unrelated 2D angle.
5.  Tag plane axes are orthogonalized (noise/perspective made the raw
    edge vectors non-orthogonal).

ROBUSTNESS / LOGIC FIXES
6.  Guard against w ~ 0 and non-finite values in triangulation (NaN used
    to silently PASS the quality filters and could emit invalid JSON).
7.  Disparity checks run FIRST, so a swapped-camera problem is reported
    as such instead of as "z out of range". A minimum disparity threshold
    was added (tiny disparity = huge depth noise).
8.  Duplicate tag IDs seen twice in one image are rejected instead of
    silently overwriting each other (which could triangulate corners from
    two DIFFERENT physical tags against each other).
9.  Left/right frame skew is measured from sensor timestamps and pairs
    with too much skew are rejected (two free-running cameras are not
    synchronized; motion during the skew creates false disparity).
10. distance_mm is computed from the geometric position only. The
    troubleshooting X_ZERO_OFFSET_MM no longer pollutes it.
11. If the capture thread dies (bad calibration file, camera failure),
    the server now publishes the error on /tag instead of serving a
    stale "server starting" forever.

MONOCHROME CAMERAS
12. Frames are captured as YUV420 and the Y (luma) plane is used
    directly as the grayscale image. No RGB conversion, no channel-order
    ambiguity, and it is faster.

ZERO OFFSETS
    X_ZERO_OFFSET_MM and Y_AXIS_ROTATION_ZERO_OFFSET_DEG are kept for
    troubleshooting but default to 0.0. After the frame fixes above they
    should stay at (or very near) 0. Sign convention for BOTH:
        put in the value the server REPORTS when the true value should
        be zero; it is subtracted from the output.
"""

import json
import time
import math
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform
from pupil_apriltags import Detector


# ============================================================
# CAMERA SETTINGS
# ============================================================

WIDTH = 1280
HEIGHT = 800
FPS = 30

LEFT_CAMERA_INDEX = 1
RIGHT_CAMERA_INDEX = 0

CALIB_FILE = "stereo_calibration.npz"

# Set True when both cameras are physically mounted upside down.
# IMPORTANT: stereo_calibration.npz must be made with the same flip setting.
FLIP_CAMERAS_180 = True

# Maximum allowed left/right capture time difference. Two free-running
# 30 fps cameras can be up to ~33 ms apart. If the rig and tag are truly
# static you can raise this or set it to None to disable the check.
MAX_FRAME_SKEW_MS = None


# ============================================================
# CALIBRATION UNITS
# ============================================================

# Units of the translation vector T from stereoCalibrate, i.e. the units
# your chessboard square size was given in during calibration.
#   "m"  -> triangulated points come out in metres
#   "mm" -> triangulated points come out in millimetres
CALIB_UNITS = "m"


# ============================================================
# HTTP SERVER SETTINGS
# ============================================================

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5005


# ============================================================
# APRILTAG SETTINGS
# ============================================================

TAG_FAMILIES = ["tag36h11", "tag25h9"]

# Optional: restrict to one family. Keep None to accept both families.
TARGET_TAG_FAMILY = None

# Real printed black-square AprilTag size in metres.
TAG_SIZE_M = 0.1

# Set to None to accept any tag, or set to a specific tag ID.
TARGET_TAG_ID = None


# ============================================================
# VISION-SIDE ZERO OFFSETS (troubleshooting only, default 0.0)
# ============================================================

# Subtracted from the reported x_mm. If a physically centered tag reads
# x_mm = +7.2, set this to +7.2. Should be ~0 after the frame fixes.
X_ZERO_OFFSET_MM = 3.0

# Subtracted from the reported tag_yaw_deg. If a physically square tag
# reads tag_yaw_deg = +3.8, set this to +3.8. Should be ~0 after the
# frame fixes.
Y_AXIS_ROTATION_ZERO_OFFSET_DEG = 0.0


# ============================================================
# QUALITY FILTERS
# ============================================================

MIN_DECISION_MARGIN = 20.0

MIN_Z_MM = 100.0
MAX_Z_MM = 2000.0

MAX_TAG_SIZE_ERROR_MM = 15.0
MAX_RECTIFIED_Y_ERROR_PX = 4.0

# Minimum disparity: below this, depth noise explodes (z ~ f*B/disparity).
MIN_DISPARITY_PX = 5.0

# If true, opens a local OpenCV preview window.
SHOW_PREVIEW = True


# ============================================================
# SHARED DATA
# ============================================================

latest_lock = threading.Lock()

latest_data = {
    "ok": False,
    "reason": "server starting",
    "source": "stereo",
    "stereo_ok": False,
    "timestamp": time.time(),
    "age_s": 0.0,
}


def publish(data):
    global latest_data
    with latest_lock:
        latest_data = data


# ============================================================
# BASIC HELPERS
# ============================================================

def minimal_error(reason, timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    return {
        "ok": False,
        "reason": str(reason),
        "source": "stereo",
        "stereo_ok": False,
        "timestamp": float(timestamp),
        "age_s": 0.0,
    }


def minimal_success(tag, timestamp, frame_skew_ms):
    return {
        "ok": True,
        "reason": None,
        "tag_id": int(tag["tag_id"]),
        "tag_family": str(tag["tag_family"]),

        "source": "stereo",
        "stereo_ok": True,

        # Position: origin = stereo midpoint, axes = PHYSICAL left camera
        # (+X right, +Y down, +Z forward).
        "x_mm": float(tag["x_mm"]),
        "y_mm": float(tag["y_mm"]),
        "z_mm": float(tag["z_mm"]),
        "distance_mm": float(tag["distance_mm"]),

        # Pixel center in the (unrectified) left image, debug only.
        "cx": float(tag["cx"]),
        "cy": float(tag["cy"]),

        "tag_pitch_deg": float(tag["tag_pitch_deg"]),
        "tag_yaw_deg": float(tag["tag_yaw_deg"]),
        "tag_roll_deg": float(tag["tag_roll_deg"]),

        # Debug: capture skew between the two cameras.
        "frame_skew_ms": None if frame_skew_ms is None else float(frame_skew_ms),

        "timestamp": float(timestamp),
        "age_s": 0.0,
    }


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration(path):
    data = np.load(path)

    calib = {
        "K_left": data["K_left"],
        "D_left": data["D_left"],
        "K_right": data["K_right"],
        "D_right": data["D_right"],
        "R1": data["R1"],
        "R2": data["R2"],
        "P1": data["P1"],
        "P2": data["P2"],
        "image_size": tuple(int(v) for v in data["image_size"]),
    }

    if CALIB_UNITS == "m":
        calib["units_to_mm"] = 1000.0
    elif CALIB_UNITS == "mm":
        calib["units_to_mm"] = 1.0
    else:
        raise RuntimeError(f"CALIB_UNITS must be 'm' or 'mm', got {CALIB_UNITS!r}")

    # FIX #2/#3: derive the baseline from P2 itself.
    # For a rectified horizontal pair, P2[0,3] = -fx * B (in calib units).
    P2 = calib["P2"]
    baseline_units = -float(P2[0, 3]) / float(P2[0, 0])
    calib["baseline_mm"] = baseline_units * calib["units_to_mm"]

    # Cross-check against a stored baseline if the file has one.
    stored_baseline_mm = None
    if "baseline_mm" in data:
        stored_baseline_mm = float(np.atleast_1d(data["baseline_mm"])[0])
    calib["stored_baseline_mm"] = stored_baseline_mm

    calib["stereo_rms"] = (
        float(np.atleast_1d(data["stereo_rms"])[0]) if "stereo_rms" in data else None
    )

    return calib


def sanity_check_calibration(calib):
    b = calib["baseline_mm"]

    print("Loaded stereo calibration:")
    print(f"  image size:            {calib['image_size']}")
    print(f"  baseline (from P2):    {b:.3f} mm  (CALIB_UNITS={CALIB_UNITS!r})")

    if calib["stored_baseline_mm"] is not None:
        sb = calib["stored_baseline_mm"]
        print(f"  baseline (stored):     {sb:.3f} mm")
        if abs(sb - b) > 0.01 * max(abs(sb), abs(b), 1.0):
            print("  WARNING: stored baseline and P2-derived baseline disagree.")
            print("           CALIB_UNITS is probably set wrong.")

    if calib["stereo_rms"] is not None:
        print(f"  stereo RMS:            {calib['stereo_rms']:.4f} px")

    # FIX #1 diagnostics: show how far the rectified frame is rotated from
    # the physical left camera. This rotation used to leak into x/yaw.
    rvec, _ = cv2.Rodrigues(calib["R1"])
    r1_deg = math.degrees(float(np.linalg.norm(rvec)))
    print(f"  R1 rectification tilt: {r1_deg:.3f} deg "
          f"(now compensated; used to appear as fake yaw / x error)")

    if not (10.0 <= b <= 1000.0):
        print("  WARNING: baseline outside 10..1000 mm. CALIB_UNITS is very "
              "likely wrong, and all outputs will be scaled incorrectly.")

    print(f"  tag size:              {TAG_SIZE_M * 1000.0:.1f} mm")
    print()

    if calib["image_size"] != (WIDTH, HEIGHT):
        print("WARNING: calibration image size does not match camera size.")
        print(f"  calibration: {calib['image_size']}")
        print(f"  camera:      {(WIDTH, HEIGHT)}")
        print()


# ============================================================
# CAMERA FUNCTIONS (monochrome)
# ============================================================

def setup_camera(index):
    cam = Picamera2(index)

    transform = Transform(hflip=1, vflip=1) if FLIP_CAMERAS_180 else Transform()

    # FIX #12: monochrome sensor. Capture YUV420 and use the Y plane as the
    # grayscale image directly. No RGB conversion, no channel-order issues.
    config = cam.create_video_configuration(
        main={
            "format": "YUV420",
            "size": (WIDTH, HEIGHT),
        },
        controls={
            "FrameRate": FPS,
        },
        transform=transform,
    )

    cam.configure(config)
    return cam


def capture_gray_with_timestamp(cam):
    """
    Capture one frame; return (gray_uint8, sensor_timestamp_ns or None).

    Uses the request API so we get the hardware SensorTimestamp, which is
    needed for the left/right skew check (FIX #9).
    """
    request = cam.capture_request()
    try:
        arr = request.make_array("main")           # YUV420: (H*3/2, W)
        metadata = request.get_metadata()
        ts_ns = metadata.get("SensorTimestamp", None)
        gray = np.ascontiguousarray(arr[:HEIGHT, :WIDTH])  # Y plane, copied
    finally:
        request.release()
    return gray, ts_ns


# ============================================================
# DETECTORS
# ============================================================

def create_detector_for_family(family):
    return Detector(
        families=family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def create_detectors():
    families_to_use = TAG_FAMILIES

    if TARGET_TAG_FAMILY is not None:
        if TARGET_TAG_FAMILY not in TAG_FAMILIES:
            raise RuntimeError(
                f"TARGET_TAG_FAMILY={TARGET_TAG_FAMILY!r} not in {TAG_FAMILIES!r}"
            )
        families_to_use = [TARGET_TAG_FAMILY]

    detectors = {}
    for family in families_to_use:
        print(f"Creating AprilTag detector for family: {family}")
        detectors[family] = create_detector_for_family(family)
    return detectors


def detections_by_family_and_id(gray_image, detectors):
    """
    Run each single-family detector; return detections keyed by (family, id).

    FIX #8: if the SAME (family, id) appears more than once in one image,
    all copies are discarded. Previously the last one silently overwrote
    the first, which could triangulate two different physical tags
    against each other.
    """
    output = {}
    duplicates = set()

    for family, detector in detectors.items():
        for det in detector.detect(gray_image):
            if TARGET_TAG_ID is not None and int(det.tag_id) != int(TARGET_TAG_ID):
                continue
            if det.decision_margin < MIN_DECISION_MARGIN:
                continue

            key = (family, int(det.tag_id))
            if key in output:
                duplicates.add(key)
            else:
                output[key] = det

    for key in duplicates:
        del output[key]

    return output


# ============================================================
# STEREO GEOMETRY
# ============================================================

def rectify_points(points, K, D, R_rect, P_rect):
    """Distorted original-image points -> rectified pixel points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    rectified = cv2.undistortPoints(points, K, D, R=R_rect, P=P_rect)
    return rectified.reshape(-1, 2)


def triangulate_tag(left_det, right_det, calib):
    """
    Triangulate the four AprilTag corners.

    Returns (corners_cam_mm, left_rect, right_rect) where corners_cam_mm is
    a 4x3 array in millimetres in the PHYSICAL left camera frame, or
    (None, left_rect, right_rect) on degenerate/non-finite triangulation.
    """
    left_rect = rectify_points(
        np.asarray(left_det.corners, dtype=np.float64),
        calib["K_left"], calib["D_left"], calib["R1"], calib["P1"],
    )
    right_rect = rectify_points(
        np.asarray(right_det.corners, dtype=np.float64),
        calib["K_right"], calib["D_right"], calib["R2"], calib["P2"],
    )

    points_4d = cv2.triangulatePoints(
        calib["P1"], calib["P2"], left_rect.T, right_rect.T,
    )

    # FIX #6: guard w ~ 0 and non-finite values. NaN used to pass filters.
    w = points_4d[3]
    if np.any(np.abs(w) < 1e-12):
        return None, left_rect, right_rect

    corners_rect = (points_4d[:3] / w).T          # rectified frame, calib units

    if not np.all(np.isfinite(corners_rect)):
        return None, left_rect, right_rect

    # FIX #1: rotate rectified-frame points back into the PHYSICAL left
    # camera frame. Rectified coords: p_rect = R1 @ p_cam  ->  p_cam = R1^T p_rect.
    corners_cam = corners_rect @ calib["R1"]      # == (R1.T @ p) per point

    # FIX #3: explicit unit conversion.
    corners_cam_mm = corners_cam * calib["units_to_mm"]

    return corners_cam_mm, left_rect, right_rect


def stereo_midpoint_offset_mm(calib):
    """
    Stereo midpoint expressed in the PHYSICAL left camera frame (mm).

    In the rectified frame the right camera sits at (B, 0, 0), so the
    midpoint is (B/2, 0, 0) there; rotate it back with R1^T (FIX #1/#2).
    """
    mid_rect = np.array([calib["baseline_mm"] / 2.0, 0.0, 0.0])
    return calib["R1"].T @ mid_rect


def estimate_tag_size_mm(corners_mm):
    p = corners_mm
    side_lengths_mm = [
        np.linalg.norm(p[1] - p[0]),
        np.linalg.norm(p[2] - p[1]),
        np.linalg.norm(p[3] - p[2]),
        np.linalg.norm(p[0] - p[3]),
    ]
    mean_size_mm = float(np.mean(side_lengths_mm))
    size_error_mm = abs(mean_size_mm - TAG_SIZE_M * 1000.0)
    return mean_size_mm, size_error_mm


def estimate_tag_orientation(corners_mm):
    """
    Orientation of the tag in the physical left camera frame.

    Convention (unchanged for the Control Pi):
      +X right, +Y down, +Z forward
      R = Ry(yaw) * Rx(pitch) * Rz(roll)
      normal straight at camera  -> yaw = 0, pitch = 0

    FIX #4/#5: axes are orthogonalized and roll is extracted by removing
    yaw and pitch first, so (yaw, pitch, roll) is now one consistent
    rotation decomposition instead of three unrelated numbers.
    """
    p = corners_mm

    # Average opposite edges for a noise-robust estimate of the tag axes.
    edge_x = ((p[1] - p[0]) + (p[2] - p[3])) / 2.0
    edge_y = ((p[3] - p[0]) + (p[2] - p[1])) / 2.0

    nx_ = np.linalg.norm(edge_x)
    ny_ = np.linalg.norm(edge_y)
    if nx_ < 1e-9 or ny_ < 1e-9:
        return None

    x_axis = edge_x / nx_

    normal = np.cross(edge_x, edge_y)
    nn = np.linalg.norm(normal)
    if nn < 1e-9:
        return None
    normal = normal / nn

    # Make the normal point away from the camera (+Z).
    if normal[2] < 0:
        normal = -normal

    # Orthogonalize x_axis against the normal (FIX #5).
    x_axis = x_axis - np.dot(x_axis, normal) * normal
    nx2 = np.linalg.norm(x_axis)
    if nx2 < 1e-9:
        return None
    x_axis = x_axis / nx2

    n_x, n_y, n_z = normal

    # Same yaw/pitch convention as before:
    #   normal = (sin(yaw)cos(pitch), -sin(pitch), cos(yaw)cos(pitch))
    yaw = math.atan2(n_x, n_z)
    pitch = math.atan2(-n_y, math.sqrt(n_x * n_x + n_z * n_z))

    # Remove yaw then pitch from x_axis; what remains is pure roll (FIX #4).
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    Ry_inv = np.array([
        [cy, 0.0, -sy],
        [0.0, 1.0, 0.0],
        [sy, 0.0, cy],
    ])
    Rx_inv = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cp, sp],
        [0.0, -sp, cp],
    ])

    x_ref = Rx_inv @ (Ry_inv @ x_axis)     # should be (cos r, sin r, ~0)
    roll = math.atan2(x_ref[1], x_ref[0])

    yaw_deg = math.degrees(yaw) - Y_AXIS_ROTATION_ZERO_OFFSET_DEG

    return {
        "tag_yaw_deg": float(yaw_deg),
        "tag_pitch_deg": float(math.degrees(pitch)),
        "tag_roll_deg": float(math.degrees(roll)),
    }


# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame_pair(left_gray, right_gray, detectors, calib, frame_skew_ms):
    timestamp = time.time()

    # FIX #9: reject unsynchronized pairs (motion -> false disparity).
    if (MAX_FRAME_SKEW_MS is not None
            and frame_skew_ms is not None
            and frame_skew_ms > MAX_FRAME_SKEW_MS):
        return minimal_error(
            f"left/right frame skew too large: {frame_skew_ms:.1f} ms", timestamp
        )

    left_by_id = detections_by_family_and_id(left_gray, detectors)
    right_by_id = detections_by_family_and_id(right_gray, detectors)

    common_ids = sorted(set(left_by_id.keys()) & set(right_by_id.keys()))
    if not common_ids:
        return minimal_error("no common AprilTag detected in both cameras", timestamp)

    mid_offset_mm = stereo_midpoint_offset_mm(calib)

    valid_tags = []
    first_reject_reason = None

    def reject(reason):
        nonlocal first_reject_reason
        if first_reject_reason is None:
            first_reject_reason = reason

    for tag_family, tag_id in common_ids:
        left_det = left_by_id[(tag_family, tag_id)]
        right_det = right_by_id[(tag_family, tag_id)]

        corners_cam_mm, left_rect, right_rect = triangulate_tag(
            left_det, right_det, calib
        )

        # FIX #7: disparity checks FIRST, so a swapped-camera or
        # too-far-away problem is reported clearly.
        disparities_px = left_rect[:, 0] - right_rect[:, 0]
        mean_disparity_px = float(np.mean(disparities_px))
        min_disparity_px = float(np.min(disparities_px))

        if mean_disparity_px <= 0:
            reject(f"bad disparity: {mean_disparity_px:.2f} px; "
                   f"left/right cameras may be swapped")
            continue

        if min_disparity_px < MIN_DISPARITY_PX:
            reject(f"disparity too small: {min_disparity_px:.2f} px "
                   f"(depth would be too noisy)")
            continue

        if corners_cam_mm is None:
            reject("degenerate triangulation (non-finite points)")
            continue

        y_errors_px = left_rect[:, 1] - right_rect[:, 1]
        mean_abs_y_err_px = float(np.mean(np.abs(y_errors_px)))
        if mean_abs_y_err_px > MAX_RECTIFIED_Y_ERROR_PX:
            reject(f"rectified y error too large: {mean_abs_y_err_px:.2f} px")
            continue

        # Geometric center: origin at stereo midpoint, physical camera axes.
        center_cam_mm = np.mean(corners_cam_mm, axis=0)
        center_mid_mm = center_cam_mm - mid_offset_mm   # FIX #1/#2

        x_mm = float(center_mid_mm[0])
        y_mm = float(center_mid_mm[1])
        z_mm = float(center_mid_mm[2])

        if z_mm < MIN_Z_MM or z_mm > MAX_Z_MM:
            reject(f"z_mm out of range: {z_mm:.1f}")
            continue

        _, tag_size_error_mm = estimate_tag_size_mm(corners_cam_mm)
        if tag_size_error_mm > MAX_TAG_SIZE_ERROR_MM:
            reject(f"tag size error too large: {tag_size_error_mm:.1f} mm")
            continue

        orientation = estimate_tag_orientation(corners_cam_mm)
        if orientation is None:
            reject("could not estimate tag orientation")
            continue

        # FIX #10: distance from geometry only, BEFORE the troubleshooting
        # x offset is applied.
        distance_mm = float(math.sqrt(x_mm * x_mm + y_mm * y_mm + z_mm * z_mm))

        valid_tags.append({
            "tag_id": int(tag_id),
            "tag_family": str(tag_family),
            "x_mm": x_mm - X_ZERO_OFFSET_MM,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "distance_mm": distance_mm,
            "cx": float(left_det.center[0]),
            "cy": float(left_det.center[1]),
            "tag_pitch_deg": orientation["tag_pitch_deg"],
            "tag_yaw_deg": orientation["tag_yaw_deg"],
            "tag_roll_deg": orientation["tag_roll_deg"],
        })

    if not valid_tags:
        return minimal_error(
            first_reject_reason or "AprilTags detected but rejected by quality checks",
            timestamp,
        )

    best = min(valid_tags, key=lambda r: r["distance_mm"])
    return minimal_success(best, timestamp, frame_skew_ms)


# ============================================================
# PREVIEW
# ============================================================

def safe_float_text(value, digits=1):
    if value is None:
        return "None"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def draw_preview(left_gray, right_gray, data):
    preview = np.hstack((left_gray, right_gray))
    preview = cv2.resize(preview, None, fx=0.5, fy=0.5)
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    if data.get("ok"):
        text = (
            f"x={data['x_mm']:.1f} y={data['y_mm']:.1f} z={data['z_mm']:.1f}mm  "
            f"yaw={safe_float_text(data.get('tag_yaw_deg'))}  "
            f"pitch={safe_float_text(data.get('tag_pitch_deg'))}  "
            f"roll={safe_float_text(data.get('tag_roll_deg'))}"
        )
        color = (0, 255, 0)
    else:
        text = str(data.get("reason", "not ok"))
        color = (0, 0, 255)

    cv2.putText(preview, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Stereo AprilTag Server: left | right", preview)
    cv2.waitKey(1)


# ============================================================
# CAPTURE THREAD
# ============================================================

def capture_loop():
    # FIX #11: any startup failure is published to /tag instead of
    # leaving the server serving "server starting" forever.
    try:
        calib = load_calibration(CALIB_FILE)
        sanity_check_calibration(calib)

        print(f"  X zero offset:          {X_ZERO_OFFSET_MM:.3f} mm")
        print(f"  Yaw zero offset:        {Y_AXIS_ROTATION_ZERO_OFFSET_DEG:.3f} deg")
        print()

        detectors = create_detectors()

        print("Starting left camera...")
        left_cam = setup_camera(LEFT_CAMERA_INDEX)
        print("Starting right camera...")
        right_cam = setup_camera(RIGHT_CAMERA_INDEX)

        left_cam.start()
        right_cam.start()
        time.sleep(1.0)

    except Exception as e:
        msg = f"startup error: {e}"
        print(msg)
        publish(minimal_error(msg))
        return

    print("Stereo AprilTag capture loop started.")

    try:
        while True:
            try:
                left_gray, left_ts = capture_gray_with_timestamp(left_cam)
                right_gray, right_ts = capture_gray_with_timestamp(right_cam)

                if left_ts is not None and right_ts is not None:
                    frame_skew_ms = abs(left_ts - right_ts) / 1e6
                else:
                    frame_skew_ms = None

                data = process_frame_pair(
                    left_gray, right_gray, detectors, calib, frame_skew_ms
                )
            except Exception as e:
                data = minimal_error(f"processing error: {e}")
                left_gray = right_gray = None

            publish(data)

            if SHOW_PREVIEW and left_gray is not None:
                draw_preview(left_gray, right_gray, data)

    finally:
        print("Stopping cameras...")
        left_cam.stop()
        right_cam.stop()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


# ============================================================
# HTTP SERVER
# ============================================================

class TagRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ["/", "/latest", "/tag", "/data"]:
            with latest_lock:
                data = dict(latest_data)

            timestamp = data.get("timestamp")
            data["age_s"] = (
                float(time.time() - float(timestamp)) if timestamp is not None else None
            )
            self._send_json(data)
            return

        if self.path == "/health":
            self._send_json({
                "ok": True,
                "server": "minimal_stereo_apriltag_server",
                "source": "stereo",
                "timestamp": time.time(),
            })
            return

        if self.path == "/config":
            self._send_json({
                "ok": True,
                "width": WIDTH,
                "height": HEIGHT,
                "fps": FPS,
                "tag_families": list(TAG_FAMILIES),
                "target_tag_family": TARGET_TAG_FAMILY,
                "tag_size_m": TAG_SIZE_M,
                "target_tag_id": TARGET_TAG_ID,
                "calib_units": CALIB_UNITS,
                "x_zero_offset_mm": X_ZERO_OFFSET_MM,
                "y_axis_rotation_zero_offset_deg": Y_AXIS_ROTATION_ZERO_OFFSET_DEG,
                "max_frame_skew_ms": MAX_FRAME_SKEW_MS,
                "min_disparity_px": MIN_DISPARITY_PX,
                "calib_file": CALIB_FILE,
                "flip_cameras_180": FLIP_CAMERAS_180,
            })
            return

        self._send_json({
            "ok": False,
            "reason": "unknown endpoint",
            "valid_endpoints": ["/", "/latest", "/tag", "/data", "/health", "/config"],
        }, status=404)

    def log_message(self, format, *args):
        return


def run_http_server():
    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), TagRequestHandler)
    print(f"HTTP server running on http://{SERVER_HOST}:{SERVER_PORT}")
    print("Useful endpoints:")
    print(f"  http://<vision-pi-ip>:{SERVER_PORT}/tag")
    print(f"  http://<vision-pi-ip>:{SERVER_PORT}/health")
    print(f"  http://<vision-pi-ip>:{SERVER_PORT}/config")
    print()
    server.serve_forever()


# ============================================================
# MAIN
# ============================================================

def main():
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    run_http_server()


if __name__ == "__main__":
    main()