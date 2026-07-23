#!/usr/bin/env python3

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2


# ------------------------------------------------------------
# Focus / image quality metrics
# ------------------------------------------------------------

def center_crop(frame, crop_fraction=0.45):
    """
    Returns the center crop and its rectangle coordinates.
    crop_fraction = 0.45 means the crop is 45% of image width/height.
    """
    h, w = frame.shape[:2]

    crop_w = int(w * crop_fraction)
    crop_h = int(h * crop_fraction)

    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def calculate_metrics(rgb_frame):
    """
    Calculates useful focus and image quality values.

    Main values:
    - laplacian_var: most useful focus value. Higher = sharper.
    - tenengrad: gradient-based sharpness. Higher = sharper.
    - brightness: average brightness. Too low/high is bad.
    - contrast: gray standard deviation. Higher usually means more visible detail.
    - clipped_dark_pct / clipped_bright_pct: percentage of under/overexposed pixels.
    """
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(sobel_x ** 2 + sobel_y ** 2)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    clipped_dark_pct = 100.0 * np.mean(gray <= 5)
    clipped_bright_pct = 100.0 * np.mean(gray >= 250)

    return {
        "laplacian_var": float(laplacian_var),
        "tenengrad": float(tenengrad),
        "brightness": brightness,
        "contrast": contrast,
        "clipped_dark_pct": float(clipped_dark_pct),
        "clipped_bright_pct": float(clipped_bright_pct),
    }


def draw_metrics(frame_bgr, title, metrics, metadata=None):
    """
    Draws text values on the image.
    """
    lines = [
        title,
        f"Laplacian: {metrics['laplacian_var']:.1f}",
        f"Tenengrad: {metrics['tenengrad']:.1f}",
        f"Brightness: {metrics['brightness']:.1f}",
        f"Contrast: {metrics['contrast']:.1f}",
        f"Dark clip: {metrics['clipped_dark_pct']:.2f}%",
        f"Bright clip: {metrics['clipped_bright_pct']:.2f}%",
    ]

    if metadata is not None:
        exposure = metadata.get("ExposureTime", None)
        gain = metadata.get("AnalogueGain", None)

        if exposure is not None:
            lines.append(f"Exposure: {exposure} us")
        if gain is not None:
            lines.append(f"Gain: {gain:.2f}")

    x = 12
    y = 28

    for i, line in enumerate(lines):
        yy = y + i * 24

        # black shadow
        cv2.putText(
            frame_bgr,
            line,
            (x + 1, yy + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        # white text
        cv2.putText(
            frame_bgr,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def make_camera(camera_index, width, height, fps):
    """
    Opens one Picamera2 camera.
    """
    cam = Picamera2(camera_index)

    frame_time_us = int(1_000_000 / fps)

    config = cam.create_video_configuration(
        main={
            "size": (width, height),
            "format": "RGB888",
        },
        controls={
            "FrameDurationLimits": (frame_time_us, frame_time_us),
        },
    )

    cam.configure(config)
    return cam


def resize_to_width(frame, target_width):
    h, w = frame.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def save_snapshot(left_rgb, right_rgb, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    left_path = os.path.join(output_dir, f"left_focus_{timestamp}.png")
    right_path = os.path.join(output_dir, f"right_focus_{timestamp}.png")

    cv2.imwrite(left_path, cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(right_path, cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved:")
    print(f"  {left_path}")
    print(f"  {right_path}")


def main():
    parser = argparse.ArgumentParser(description="Stereo camera focus calibration helper")

    parser.add_argument("--width", type=int, default=640, help="Preview width per camera")
    parser.add_argument("--height", type=int, default=480, help="Preview height per camera")
    parser.add_argument("--fps", type=int, default=20, help="Camera FPS")
    parser.add_argument("--crop", type=float, default=0.45, help="Center crop fraction for focus metrics")
    parser.add_argument("--left", type=int, default=0, help="Left camera index")
    parser.add_argument("--right", type=int, default=1, help="Right camera index")
    parser.add_argument("--log", action="store_true", help="Log focus metrics to CSV")
    parser.add_argument("--output-dir", default="focus_output", help="Folder for snapshots/logs")

    args = parser.parse_args()

    print()
    print("Available cameras:")
    for i, info in enumerate(Picamera2.global_camera_info()):
        print(f"  Camera {i}: {info}")
    print()

    if len(Picamera2.global_camera_info()) < 2:
        print("ERROR: Less than two cameras detected.")
        print("Check ribbon cables, camera connectors, and run:")
        print("  libcamera-hello --list-cameras")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    csv_file = None
    csv_writer = None

    if args.log:
        csv_path = os.path.join(
            args.output_dir,
            "focus_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv",
        )

        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            "timestamp",
            "left_laplacian",
            "right_laplacian",
            "left_tenengrad",
            "right_tenengrad",
            "left_brightness",
            "right_brightness",
            "left_contrast",
            "right_contrast",
            "left_dark_clip_pct",
            "right_dark_clip_pct",
            "left_bright_clip_pct",
            "right_bright_clip_pct",
            "left_exposure_us",
            "right_exposure_us",
            "left_gain",
            "right_gain",
        ])

        print(f"Logging to: {csv_path}")

    print("Starting cameras...")
    left_cam = make_camera(args.left, args.width, args.height, args.fps)
    right_cam = make_camera(args.right, args.width, args.height, args.fps)

    left_cam.start()
    right_cam.start()

    time.sleep(1.0)

    print()
    print("Controls:")
    print("  q  = quit")
    print("  s  = save left/right snapshot")
    print()
    print("Focus method:")
    print("  1. Put AprilTag/chessboard at your real working distance, e.g. 500 mm.")
    print("  2. Adjust each lens until its Laplacian value reaches its own maximum.")
    print("  3. Do not force both cameras to have exactly the same number.")
    print("  4. After focusing, redo stereo calibration.")
    print()

    try:
        while True:
            left_rgb = left_cam.capture_array()
            right_rgb = right_cam.capture_array()

            left_metadata = left_cam.capture_metadata()
            right_metadata = right_cam.capture_metadata()

            left_crop_rgb, left_rect = center_crop(left_rgb, args.crop)
            right_crop_rgb, right_rect = center_crop(right_rgb, args.crop)

            left_metrics = calculate_metrics(left_crop_rgb)
            right_metrics = calculate_metrics(right_crop_rgb)

            left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
            right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)

            lx1, ly1, lx2, ly2 = left_rect
            rx1, ry1, rx2, ry2 = right_rect

            cv2.rectangle(left_bgr, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
            cv2.rectangle(right_bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

            draw_metrics(left_bgr, "LEFT CAMERA", left_metrics, left_metadata)
            draw_metrics(right_bgr, "RIGHT CAMERA", right_metrics, right_metadata)

            left_crop_bgr = cv2.cvtColor(left_crop_rgb, cv2.COLOR_RGB2BGR)
            right_crop_bgr = cv2.cvtColor(right_crop_rgb, cv2.COLOR_RGB2BGR)

            left_crop_zoom = cv2.resize(
                left_crop_bgr,
                (args.width, args.height),
                interpolation=cv2.INTER_NEAREST,
            )

            right_crop_zoom = cv2.resize(
                right_crop_bgr,
                (args.width, args.height),
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.putText(
                left_crop_zoom,
                "LEFT CENTER CROP - focus here",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                right_crop_zoom,
                "RIGHT CENTER CROP - focus here",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            top = np.hstack([left_bgr, right_bgr])
            bottom = np.hstack([left_crop_zoom, right_crop_zoom])
            combined = np.vstack([top, bottom])

            cv2.imshow("Stereo Focus Calibrator", combined)

            if csv_writer is not None:
                csv_writer.writerow([
                    datetime.now().isoformat(),
                    left_metrics["laplacian_var"],
                    right_metrics["laplacian_var"],
                    left_metrics["tenengrad"],
                    right_metrics["tenengrad"],
                    left_metrics["brightness"],
                    right_metrics["brightness"],
                    left_metrics["contrast"],
                    right_metrics["contrast"],
                    left_metrics["clipped_dark_pct"],
                    right_metrics["clipped_dark_pct"],
                    left_metrics["clipped_bright_pct"],
                    right_metrics["clipped_bright_pct"],
                    left_metadata.get("ExposureTime", ""),
                    right_metadata.get("ExposureTime", ""),
                    left_metadata.get("AnalogueGain", ""),
                    right_metadata.get("AnalogueGain", ""),
                ])

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                save_snapshot(left_rgb, right_rgb, args.output_dir)

    finally:
        print("Stopping cameras...")

        left_cam.stop()
        right_cam.stop()

        if csv_file is not None:
            csv_file.close()

        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
