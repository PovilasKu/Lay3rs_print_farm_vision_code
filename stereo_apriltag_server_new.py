#!/usr/bin/env python3

"""
Stereo AprilTag server for the Vision Pi.

Changes in this version:
- Keeps the original stereo triangulation approach.
- Adds explicit tag_yaw_deg and tag_pitch_deg outputs, while keeping the old
  tag_tilt_x_deg/tag_tilt_y_deg/tag_roll_deg keys for compatibility.
- Adds stereo/source flags expected by the Control Pi.
- Adds a frame_id and more quality fields for debugging.
- Keeps /latest, /tag, /data and /health endpoints.
"""

import json
import time
import math
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
from picamera2 import Picamera2
from pupil_apriltags import Detector


# ============================================================
# CAMERA SETTINGS
# ============================================================

WIDTH = 1280
HEIGHT = 800
FPS = 30

LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 1

CALIB_FILE = "stereo_calibration.npz"


# ============================================================
# HTTP SERVER SETTINGS
# ============================================================

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5005


# ============================================================
# APRILTAG SETTINGS
# ============================================================

TAG_FAMILY = "tag36h11"

# IMPORTANT: set this to your real printed black-square AprilTag size.
# 65 mm tag  -> 0.065
# 80 mm tag  -> 0.080
# 100 mm tag -> 0.100
TAG_SIZE_M = 0.1

# Set to None to accept any tag.
# Set to a number, for example 0, to only use tag ID 0.
TARGET_TAG_ID = None


# ============================================================
# STEREO REFERENCE POINT
# ============================================================

# Your cameras are 100 mm apart, so the midpoint is 50 mm
# to the right of the left camera origin.
#
# The raw triangulation gives x relative to the left camera.
# We output:
#
#   x_mm = x_leftcam_mm - 50.0
#
# Therefore x_mm = 0 means the tag is centered between the two cameras.
CENTER_X_OFFSET_MM = 50.0


# ============================================================
# QUALITY FILTERS
# ============================================================

MIN_DECISION_MARGIN = 20.0

MIN_Z_MM = 100.0
MAX_Z_MM = 2000.0

# This checks whether the triangulated 3D AprilTag size agrees
# with your real tag size. Increase if needed after checking data.
MAX_TAG_SIZE_ERROR_MM = 15.0

# If rectified left/right points do not have similar y-values,
# stereo geometry is probably bad for this frame.
MAX_RECTIFIED_Y_ERROR_PX = 4.0

# If true, opens a local OpenCV preview window.
# Set to False if running headless over SSH.
SHOW_PREVIEW = False


# ============================================================
# SHARED DATA
# ============================================================

latest_lock = threading.Lock()
frame_lock = threading.Lock()
frame_id = 0

latest_data = {
    "ok": False,
    "reason": "server starting",
    "timestamp": time.time(),
    "age_s": 0.0,
    "source": "stereo",
    "stereo_ok": False,
}


# ============================================================
# CALIBRATION / CAMERA FUNCTIONS
# ============================================================

def next_frame_id():
    global frame_id
    with frame_lock:
        frame_id += 1
        return frame_id


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

    if "baseline_mm" in data:
        calib["baseline_mm"] = float(data["baseline_mm"][0])
    else:
        calib["baseline_mm"] = None

    if "stereo_rms" in data:
        calib["stereo_rms"] = float(data["stereo_rms"][0])
    else:
        calib["stereo_rms"] = None

    return calib


def setup_camera(index):
    cam = Picamera2(index)

    config = cam.create_video_configuration(
        main={
            "format": "RGB888",
            "size": (WIDTH, HEIGHT),
        },
        controls={
            "FrameRate": FPS,
        },
    )

    cam.configure(config)
    return cam


def create_detector():
    return Detector(
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


# ============================================================
# STEREO GEOMETRY FUNCTIONS
# ============================================================

def rectify_points(points, K, D, R_rect, P_rect):
    """
    Convert distorted original image points into rectified pixel points.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)

    rectified = cv2.undistortPoints(
        points,
        K,
        D,
        R=R_rect,
        P=P_rect,
    )

    return rectified.reshape(-1, 2)


def detections_by_id(detections):
    """
    Keep only detections that pass ID and margin checks.
    """
    output = {}

    for det in detections:
        if TARGET_TAG_ID is not None and det.tag_id != TARGET_TAG_ID:
            continue

        if det.decision_margin < MIN_DECISION_MARGIN:
            continue

        output[int(det.tag_id)] = det

    return output


def triangulate_tag(left_det, right_det, calib):
    """
    Triangulates the four AprilTag corners.

    Returns 3D points in metres in the left rectified camera frame.
    """
    left_corners = np.asarray(left_det.corners, dtype=np.float64)
    right_corners = np.asarray(right_det.corners, dtype=np.float64)

    left_rect = rectify_points(
        left_corners,
        calib["K_left"],
        calib["D_left"],
        calib["R1"],
        calib["P1"],
    )

    right_rect = rectify_points(
        right_corners,
        calib["K_right"],
        calib["D_right"],
        calib["R2"],
        calib["P2"],
    )

    points_4d = cv2.triangulatePoints(
        calib["P1"],
        calib["P2"],
        left_rect.T,
        right_rect.T,
    )

    points_3d_m = (points_4d[:3] / points_4d[3]).T

    return points_3d_m, left_rect, right_rect


def estimate_tag_size_mm(corners_3d_m):
    p = corners_3d_m

    side_lengths_m = [
        np.linalg.norm(p[1] - p[0]),
        np.linalg.norm(p[2] - p[1]),
        np.linalg.norm(p[3] - p[2]),
        np.linalg.norm(p[0] - p[3]),
    ]

    mean_size_m = float(np.mean(side_lengths_m))
    mean_size_mm = mean_size_m * 1000.0
    size_error_mm = abs(mean_size_mm - TAG_SIZE_M * 1000.0)

    return mean_size_mm, size_error_mm


def estimate_tag_orientation(corners_3d_m):
    """
    Estimate tag orientation from triangulated 3D corners.

    Coordinate convention:
      +X = right in image/stereo frame
      +Y = down in image/stereo frame
      +Z = forward from cameras

    Explicit outputs:
      tag_yaw_deg   = left/right plane yaw from the tag normal
      tag_pitch_deg = up/down plane pitch from the tag normal
      tag_roll_deg  = in-plane image/tag roll

    Compatibility outputs:
      tag_tilt_x_deg = old pitch-like value
      tag_tilt_y_deg = old yaw-like value, sign may differ from tag_yaw_deg
    """
    p = corners_3d_m

    edge_x = p[1] - p[0]
    edge_y = p[3] - p[0]

    norm_x = np.linalg.norm(edge_x)
    norm_y = np.linalg.norm(edge_y)

    if norm_x < 1e-9 or norm_y < 1e-9:
        return None

    x_axis = edge_x / norm_x
    y_axis = edge_y / norm_y

    normal = np.cross(x_axis, y_axis)
    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-9:
        return None

    normal = normal / normal_norm

    # Make normal point generally forward.
    if normal[2] < 0:
        normal = -normal

    nx, ny, nz = normal

    # More explicit names for control code.
    tag_yaw_deg = math.degrees(math.atan2(nx, nz))
    tag_pitch_deg = math.degrees(math.atan2(-ny, math.sqrt(nx * nx + nz * nz)))

    # Legacy fields retained for older control files.
    tag_tilt_x_deg = math.degrees(math.atan2(ny, nz))
    tag_tilt_y_deg = math.degrees(math.atan2(-nx, nz))

    # Roll from tag horizontal edge in the camera image/stereo plane.
    tag_roll_deg = math.degrees(math.atan2(x_axis[1], x_axis[0]))

    return {
        "tag_yaw_deg": float(tag_yaw_deg),
        "tag_pitch_deg": float(tag_pitch_deg),
        "tag_roll_deg": float(tag_roll_deg),

        # Compatibility with the older control script.
        "tag_tilt_x_deg": float(tag_tilt_x_deg),
        "tag_tilt_y_deg": float(tag_tilt_y_deg),

        "normal_x": float(nx),
        "normal_y": float(ny),
        "normal_z": float(nz),
        "tag_x_axis_x": float(x_axis[0]),
        "tag_x_axis_y": float(x_axis[1]),
        "tag_x_axis_z": float(x_axis[2]),
    }


def process_frame_pair(left_rgb, right_rgb, detector, calib):
    timestamp = time.time()
    this_frame_id = next_frame_id()

    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)

    left_dets = detector.detect(left_gray)
    right_dets = detector.detect(right_gray)

    left_by_id = detections_by_id(left_dets)
    right_by_id = detections_by_id(right_dets)

    common_ids = sorted(set(left_by_id.keys()) & set(right_by_id.keys()))

    if not common_ids:
        return {
            "ok": False,
            "reason": "no common AprilTag detected in both cameras",
            "timestamp": timestamp,
            "age_s": 0.0,
            "frame_id": this_frame_id,
            "source": "stereo",
            "left_ok": bool(left_by_id),
            "right_ok": bool(right_by_id),
            "stereo_ok": False,
            "left_detected_ids": sorted(list(left_by_id.keys())),
            "right_detected_ids": sorted(list(right_by_id.keys())),
            "common_ids": common_ids,
        }

    tag_results = []

    for tag_id in common_ids:
        left_det = left_by_id[tag_id]
        right_det = right_by_id[tag_id]

        corners_3d_m, left_rect, right_rect = triangulate_tag(
            left_det,
            right_det,
            calib,
        )

        center_leftcam_m = np.mean(corners_3d_m, axis=0)
        center_leftcam_mm = center_leftcam_m * 1000.0

        x_leftcam_mm = float(center_leftcam_mm[0])
        y_leftcam_mm = float(center_leftcam_mm[1])
        z_leftcam_mm = float(center_leftcam_mm[2])

        # Convert from left-camera reference to stereo-center reference.
        x_center_mm = x_leftcam_mm - CENTER_X_OFFSET_MM
        y_center_mm = y_leftcam_mm
        z_center_mm = z_leftcam_mm

        distance_center_mm = float(
            math.sqrt(
                x_center_mm * x_center_mm
                + y_center_mm * y_center_mm
                + z_center_mm * z_center_mm
            )
        )

        distance_leftcam_mm = float(
            math.sqrt(
                x_leftcam_mm * x_leftcam_mm
                + y_leftcam_mm * y_leftcam_mm
                + z_leftcam_mm * z_leftcam_mm
            )
        )

        measured_tag_size_mm, tag_size_error_mm = estimate_tag_size_mm(corners_3d_m)

        y_errors_px = left_rect[:, 1] - right_rect[:, 1]
        mean_abs_rectified_y_error_px = float(np.mean(np.abs(y_errors_px)))
        max_abs_rectified_y_error_px = float(np.max(np.abs(y_errors_px)))

        disparities_px = left_rect[:, 0] - right_rect[:, 0]
        mean_disparity_px = float(np.mean(disparities_px))

        orientation = estimate_tag_orientation(corners_3d_m)

        ok = True
        reject_reason = None

        if z_center_mm < MIN_Z_MM or z_center_mm > MAX_Z_MM:
            ok = False
            reject_reason = f"z_mm out of range: {z_center_mm:.1f}"

        if tag_size_error_mm > MAX_TAG_SIZE_ERROR_MM:
            ok = False
            reject_reason = f"tag size error too large: {tag_size_error_mm:.1f} mm"

        if mean_abs_rectified_y_error_px > MAX_RECTIFIED_Y_ERROR_PX:
            ok = False
            reject_reason = (
                f"rectified y error too large: "
                f"{mean_abs_rectified_y_error_px:.2f} px"
            )

        if mean_disparity_px <= 0:
            ok = False
            reject_reason = (
                f"bad disparity: {mean_disparity_px:.2f} px; "
                "left/right cameras may be swapped"
            )

        result = {
            "ok": ok,
            "reject_reason": reject_reason,

            "tag_id": int(tag_id),

            "source": "stereo",
            "left_ok": True,
            "right_ok": True,
            "stereo_ok": bool(ok),

            # Main output, measured from center between cameras.
            "reference_frame": "stereo_center_between_cameras",
            "center_x_offset_mm": CENTER_X_OFFSET_MM,

            "x_mm": float(x_center_mm),
            "y_mm": float(y_center_mm),
            "z_mm": float(z_center_mm),
            "distance_mm": float(distance_center_mm),

            # Extra raw left-camera coordinates for debugging.
            "x_leftcam_mm": float(x_leftcam_mm),
            "y_leftcam_mm": float(y_leftcam_mm),
            "z_leftcam_mm": float(z_leftcam_mm),
            "distance_leftcam_mm": float(distance_leftcam_mm),

            # Pixel information.
            "cx": float(left_det.center[0]),
            "cy": float(left_det.center[1]),
            "cx_left": float(left_det.center[0]),
            "cy_left": float(left_det.center[1]),
            "cx_right": float(right_det.center[0]),
            "cy_right": float(right_det.center[1]),

            # Stereo quality information.
            "mean_disparity_px": float(mean_disparity_px),
            "disparity_px": float(mean_disparity_px),
            "rectified_y_error_px": float(mean_abs_rectified_y_error_px),
            "rectified_y_error_max_px": float(max_abs_rectified_y_error_px),

            # Tag-size quality check.
            "tag_size_expected_mm": float(TAG_SIZE_M * 1000.0),
            "tag_size_measured_mm": float(measured_tag_size_mm),
            "tag_size_error_mm": float(tag_size_error_mm),

            # Detector confidence.
            "left_margin": float(left_det.decision_margin),
            "right_margin": float(right_det.decision_margin),
            "min_decision_margin": float(min(left_det.decision_margin, right_det.decision_margin)),

            # Time.
            "timestamp": timestamp,
            "age_s": 0.0,
            "frame_id": this_frame_id,
        }

        if orientation is not None:
            result.update(orientation)
        else:
            result.update({
                "tag_yaw_deg": None,
                "tag_pitch_deg": None,
                "tag_roll_deg": None,
                "tag_tilt_x_deg": None,
                "tag_tilt_y_deg": None,
            })

        tag_results.append(result)

    valid_tags = [r for r in tag_results if r["ok"]]

    if not valid_tags:
        return {
            "ok": False,
            "reason": "AprilTags detected but rejected by quality checks",
            "timestamp": timestamp,
            "age_s": 0.0,
            "frame_id": this_frame_id,
            "source": "stereo",
            "left_ok": True,
            "right_ok": True,
            "stereo_ok": False,
            "common_ids": common_ids,
            "tags": tag_results,
        }

    # If multiple tags are visible, use the closest valid one.
    best = sorted(valid_tags, key=lambda r: r["distance_mm"])[0]

    # Main JSON output keeps old-style direct fields at top level.
    output = {
        "ok": True,
        "reason": None,

        "tag_id": best["tag_id"],

        "source": "stereo",
        "left_ok": True,
        "right_ok": True,
        "stereo_ok": True,

        "reference_frame": best["reference_frame"],
        "center_x_offset_mm": best["center_x_offset_mm"],

        "x_mm": best["x_mm"],
        "y_mm": best["y_mm"],
        "z_mm": best["z_mm"],
        "distance_mm": best["distance_mm"],

        "x_leftcam_mm": best["x_leftcam_mm"],
        "y_leftcam_mm": best["y_leftcam_mm"],
        "z_leftcam_mm": best["z_leftcam_mm"],
        "distance_leftcam_mm": best["distance_leftcam_mm"],

        "cx": best["cx"],
        "cy": best["cy"],
        "cx_left": best["cx_left"],
        "cy_left": best["cy_left"],
        "cx_right": best["cx_right"],
        "cy_right": best["cy_right"],

        # Explicit orientation fields for the new controller.
        "tag_yaw_deg": best.get("tag_yaw_deg"),
        "tag_pitch_deg": best.get("tag_pitch_deg"),
        "tag_roll_deg": best.get("tag_roll_deg"),

        # Legacy fields for older controller compatibility.
        "tag_tilt_x_deg": best.get("tag_tilt_x_deg"),
        "tag_tilt_y_deg": best.get("tag_tilt_y_deg"),

        "normal_x": best.get("normal_x"),
        "normal_y": best.get("normal_y"),
        "normal_z": best.get("normal_z"),

        "mean_disparity_px": best["mean_disparity_px"],
        "disparity_px": best["disparity_px"],
        "rectified_y_error_px": best["rectified_y_error_px"],
        "rectified_y_error_max_px": best["rectified_y_error_max_px"],

        "tag_size_expected_mm": best["tag_size_expected_mm"],
        "tag_size_measured_mm": best["tag_size_measured_mm"],
        "tag_size_error_mm": best["tag_size_error_mm"],

        "left_margin": best["left_margin"],
        "right_margin": best["right_margin"],
        "min_decision_margin": best["min_decision_margin"],

        "timestamp": best["timestamp"],
        "age_s": 0.0,
        "frame_id": best["frame_id"],

        "all_tags": tag_results,
    }

    return output


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


def draw_preview(left_rgb, right_rgb, data):
    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)

    preview = np.hstack((left_bgr, right_bgr))
    preview = cv2.resize(preview, None, fx=0.5, fy=0.5)

    if data.get("ok"):
        text = (
            f"x={data['x_mm']:.1f} y={data['y_mm']:.1f} z={data['z_mm']:.1f}mm  "
            f"yaw={safe_float_text(data.get('tag_yaw_deg'), 1)}  "
            f"pitch={safe_float_text(data.get('tag_pitch_deg'), 1)}  "
            f"roll={safe_float_text(data.get('tag_roll_deg'), 1)}"
        )
        color = (0, 255, 0)
    else:
        text = data.get("reason", "not ok")
        color = (0, 0, 255)

    cv2.putText(
        preview,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.imshow("Stereo AprilTag Server: left | right", preview)
    cv2.waitKey(1)


# ============================================================
# CAPTURE THREAD
# ============================================================

def capture_loop():
    global latest_data

    calib = load_calibration(CALIB_FILE)

    print("Loaded stereo calibration:")
    print(f"  image size: {calib['image_size']}")

    if calib["baseline_mm"] is not None:
        print(f"  calibrated baseline: {calib['baseline_mm']:.3f} mm")

    if calib["stereo_rms"] is not None:
        print(f"  stereo RMS: {calib['stereo_rms']:.4f} px")

    print(f"  output reference: stereo center, x offset = {CENTER_X_OFFSET_MM:.1f} mm")
    print(f"  tag size: {TAG_SIZE_M * 1000.0:.1f} mm")
    print()

    if calib["image_size"] != (WIDTH, HEIGHT):
        print("WARNING: calibration image size does not match camera size.")
        print(f"  calibration: {calib['image_size']}")
        print(f"  camera:      {(WIDTH, HEIGHT)}")
        print()

    detector = create_detector()

    print("Starting left camera...")
    left_cam = setup_camera(LEFT_CAMERA_INDEX)

    print("Starting right camera...")
    right_cam = setup_camera(RIGHT_CAMERA_INDEX)

    left_cam.start()
    right_cam.start()

    time.sleep(1.0)

    print("Stereo AprilTag capture loop started.")

    try:
        while True:
            left_rgb = left_cam.capture_array()
            right_rgb = right_cam.capture_array()

            try:
                data = process_frame_pair(left_rgb, right_rgb, detector, calib)
            except Exception as e:
                data = {
                    "ok": False,
                    "reason": f"processing error: {e}",
                    "timestamp": time.time(),
                    "age_s": 0.0,
                    "frame_id": next_frame_id(),
                    "source": "stereo",
                    "stereo_ok": False,
                }

            with latest_lock:
                latest_data = data

            if SHOW_PREVIEW:
                draw_preview(left_rgb, right_rgb, data)

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

            now = time.time()
            timestamp = data.get("timestamp")

            if timestamp is not None:
                data["age_s"] = float(now - timestamp)
            else:
                data["age_s"] = None

            self._send_json(data)
            return

        if self.path == "/health":
            self._send_json({
                "ok": True,
                "server": "stereo_apriltag_server_precise",
                "timestamp": time.time(),
                "source": "stereo",
            })
            return

        if self.path == "/config":
            self._send_json({
                "ok": True,
                "width": WIDTH,
                "height": HEIGHT,
                "fps": FPS,
                "tag_family": TAG_FAMILY,
                "tag_size_m": TAG_SIZE_M,
                "target_tag_id": TARGET_TAG_ID,
                "center_x_offset_mm": CENTER_X_OFFSET_MM,
                "calib_file": CALIB_FILE,
            })
            return

        self._send_json({
            "ok": False,
            "reason": "unknown endpoint",
            "valid_endpoints": ["/", "/latest", "/tag", "/data", "/health", "/config"],
        }, status=404)

    def log_message(self, format, *args):
        # Silence normal HTTP logging.
        return


def run_http_server():
    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), TagRequestHandler)

    print(f"HTTP server running on http://{SERVER_HOST}:{SERVER_PORT}")
    print("Useful endpoints:")
    print(f"  http://<vision-pi-ip>:{SERVER_PORT}/latest")
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
