#!/usr/bin/env python3

import argparse
import glob
import os
import cv2
import numpy as np


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
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )

    return cv2.aruco.CharucoBoard_create(
        SQUARES_X,
        SQUARES_Y,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )


def get_board_chessboard_corners(board):
    if hasattr(board, "getChessboardCorners"):
        return board.getChessboardCorners()
    return board.chessboardCorners


def detect_charuco(gray, dictionary, board):
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

    if marker_ids is None or len(marker_ids) == 0:
        return None, None

    count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )

    if charuco_ids is None or count < MIN_CHARUCO_CORNERS:
        return None, None

    return charuco_corners.astype(np.float32), charuco_ids.astype(np.int32)


def charuco_to_calibration_points(charuco_corners, charuco_ids, board):
    board_corners = get_board_chessboard_corners(board)

    ids = charuco_ids.flatten().astype(int)

    object_points = board_corners[ids].astype(np.float32)
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)

    object_points = object_points.reshape(-1, 1, 3)
    image_points = image_points.reshape(-1, 1, 2)

    return object_points, image_points


def reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error_sq = 0.0
    total_points = 0

    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )

        error = cv2.norm(img, projected, cv2.NORM_L2)
        total_error_sq += error * error
        total_points += len(obj)

    if total_points == 0:
        return float("inf")

    return float(np.sqrt(total_error_sq / total_points))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing calibration PNG images")
    parser.add_argument("--output", required=True, help="Output .npz calibration file")
    parser.add_argument("--name", default="camera", help="Camera name, e.g. left or right")
    args = parser.parse_args()

    dictionary = make_dictionary()
    board = make_board(dictionary)

    image_paths = sorted(glob.glob(os.path.join(args.input, "*.png")))

    if not image_paths:
        raise RuntimeError(f"No PNG images found in {args.input}")

    all_object_points = []
    all_image_points = []
    used_images = []

    image_size = None

    for path in image_paths:
        image = cv2.imread(path)

        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            h, w = gray.shape[:2]
            image_size = (w, h)

        charuco_corners, charuco_ids = detect_charuco(gray, dictionary, board)

        if charuco_corners is None:
            print(f"Skipping {path}: not enough ChArUco corners")
            continue

        obj_pts, img_pts = charuco_to_calibration_points(
            charuco_corners,
            charuco_ids,
            board,
        )

        all_object_points.append(obj_pts)
        all_image_points.append(img_pts)
        used_images.append(path)

        print(f"Using {path}: {len(img_pts)} corners")

    if len(all_object_points) < 15:
        raise RuntimeError(
            f"Only {len(all_object_points)} usable images. "
            "Capture more varied calibration images."
        )

    print()
    print(f"Calibrating {args.name} camera using {len(all_object_points)} images...")
    print(f"Image size: {image_size}")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,
        None,
    )

    mean_error = reprojection_error(
        all_object_points,
        all_image_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    print()
    print("Calibration complete.")
    print(f"Camera: {args.name}")
    print(f"OpenCV RMS error: {rms:.4f} px")
    print(f"Mean reprojection error: {mean_error:.4f} px")
    print()
    print("Camera matrix K:")
    print(camera_matrix)
    print()
    print("Distortion coefficients D:")
    print(dist_coeffs.ravel())
    print()

    np.savez(
        args.output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size, dtype=np.int32),
        rms=np.array([rms], dtype=np.float64),
        mean_reprojection_error_px=np.array([mean_error], dtype=np.float64),
        squares_x=np.array([SQUARES_X], dtype=np.int32),
        squares_y=np.array([SQUARES_Y], dtype=np.int32),
        square_length_m=np.array([SQUARE_LENGTH_M], dtype=np.float64),
        marker_length_m=np.array([MARKER_LENGTH_M], dtype=np.float64),
        used_images=np.array(used_images),
    )

    print(f"Saved calibration to: {args.output}")


if __name__ == "__main__":
    main()
