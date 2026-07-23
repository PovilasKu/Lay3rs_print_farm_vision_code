#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform

FLIP_CAMERAS_180 = True


# ----------------------------
# CAMERA SETTINGS
# ----------------------------
WIDTH = 1280
HEIGHT = 800
FPS = 30

LEFT_CAMERA_INDEX = 1
RIGHT_CAMERA_INDEX = 0

SAVE_DIR = "stereo_calib_images"
LEFT_DIR = os.path.join(SAVE_DIR, "left")
RIGHT_DIR = os.path.join(SAVE_DIR, "right")


# ----------------------------
# YOUR CHARUCO BOARD SETTINGS
# ----------------------------
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH_M = 0.035
MARKER_LENGTH_M = 0.022
MIN_CHARUCO_CORNERS = 8


def make_dictionary():
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)


def make_board(dictionary):
    # Newer OpenCV API
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )

    # Older OpenCV API
    return cv2.aruco.CharucoBoard_create(
        SQUARES_X,
        SQUARES_Y,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )


def detect_charuco(image_rgb, dictionary, board):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    charuco_corners = None
    charuco_ids = None
    count = 0

    if marker_ids is not None and len(marker_ids) > 0:
        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )

    return marker_corners, marker_ids, charuco_corners, charuco_ids, int(count)


def draw_detection(image_rgb, dictionary, board):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    marker_corners, marker_ids, charuco_corners, charuco_ids, count = detect_charuco(
        image_rgb,
        dictionary,
        board,
    )

    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(image_bgr, marker_corners, marker_ids)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(
            image_bgr,
            charuco_corners,
            charuco_ids,
            (0, 255, 0),
        )

    return image_bgr, count


def setup_camera(index):
    cam = Picamera2(index)

    transform = Transform(hflip=1, vflip=1) if FLIP_CAMERAS_180 else Transform()

    config = cam.create_video_configuration(
        main={
            "format": "RGB888",
            "size": (WIDTH, HEIGHT),
        },
        controls={
            "FrameRate": FPS,
        },
        transform=transform,
    )

    cam.configure(config)
    return cam


def next_index():
    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    existing = [
        f for f in os.listdir(LEFT_DIR)
        if f.lower().endswith(".png")
    ]

    if not existing:
        return 0

    nums = []
    for f in existing:
        name = os.path.splitext(f)[0]
        try:
            nums.append(int(name))
        except ValueError:
            pass

    if not nums:
        return 0

    return max(nums) + 1


def main():
    dictionary = make_dictionary()
    board = make_board(dictionary)

    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    print("Starting left camera...")
    left_cam = setup_camera(LEFT_CAMERA_INDEX)

    print("Starting right camera...")
    right_cam = setup_camera(RIGHT_CAMERA_INDEX)

    left_cam.start()
    right_cam.start()

    time.sleep(1.0)

    img_index = next_index()

    print()
    print("Controls:")
    print("  SPACE or s  = save stereo pair")
    print("  q           = quit")
    print()
    print("Only save when both cameras show enough detected ChArUco corners.")
    print()

    try:
        while True:
            left_rgb = left_cam.capture_array()
            right_rgb = right_cam.capture_array()

            left_vis, left_count = draw_detection(left_rgb, dictionary, board)
            right_vis, right_count = draw_detection(right_rgb, dictionary, board)

            left_ok = left_count >= MIN_CHARUCO_CORNERS
            right_ok = right_count >= MIN_CHARUCO_CORNERS

            status = f"LEFT corners: {left_count}  RIGHT corners: {right_count}"
            ok_text = "SAVE OK" if left_ok and right_ok else "MOVE BOARD / NEED MORE CORNERS"

            cv2.putText(
                left_vis,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if left_ok else (0, 0, 255),
                2,
            )

            cv2.putText(
                right_vis,
                ok_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if right_ok else (0, 0, 255),
                2,
            )

            preview = np.hstack((left_vis, right_vis))

            # Resize preview so it fits on screen.
            preview_small = cv2.resize(
                preview,
                None,
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow("Stereo ChArUco capture: left | right", preview_small)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" ") or key == ord("s"):
                if not (left_ok and right_ok):
                    print(
                        f"Not saved. Need at least {MIN_CHARUCO_CORNERS} corners "
                        f"in both images. Left={left_count}, Right={right_count}"
                    )
                    continue

                left_path = os.path.join(LEFT_DIR, f"{img_index:03d}.png")
                right_path = os.path.join(RIGHT_DIR, f"{img_index:03d}.png")

                # Save as BGR-compatible PNGs.
                cv2.imwrite(left_path, cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(right_path, cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))

                print(f"Saved pair {img_index:03d}:")
                print(f"  {left_path}")
                print(f"  {right_path}")

                img_index += 1

    finally:
        print("Stopping cameras...")
        left_cam.stop()
        right_cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
