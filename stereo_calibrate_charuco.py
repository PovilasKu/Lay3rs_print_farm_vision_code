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
MIN_COMMON_CORNERS = 8


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

    if charuco_ids is None or count < MIN_COMMON_CORNERS:
        return None, None

    return charuco_corners.astype(np.float32), charuco_ids.astype(np.int32)


def common_charuco_points(left_corners, left_ids, right_corners, right_ids, board):
    board_corners = get_board_chessboard_corners(board)

    left_ids_flat = left_ids.flatten().astype(int)
    right_ids_flat = right_ids.flatten().astype(int)

    left_points = {
        int(i): p for i, p in zip(left_ids_flat, left_corners.reshape(-1, 2))
    }

    right_points = {
        int(i): p for i, p in zip(right_ids_flat, right_corners.reshape(-1, 2))
    }

    common_ids = sorted(set(left_points.keys()) & set(right_points.keys()))

    if len(common_ids) < MIN_COMMON_CORNERS:
        return None, None, None, []

    object_points = np.array(
        [board_corners[i] for i in common_ids],
        dtype=np.float32,
    ).reshape(-1, 1, 3)

    image_points_left = np.array(
        [left_points[i] for i in common_ids],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    image_points_right = np.array(
        [right_points[i] for i in common_ids],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    return object_points, image_points_left, image_points_right, common_ids


def load_calibration(path):
    data = np.load(path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    image_size = tuple(int(v) for v in data["image_size"])
    return camera_matrix, dist_coeffs, image_size


def paired_image_paths(left_dir, right_dir):
    left_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(left_dir, "*.png"))
    }

    right_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(right_dir, "*.png"))
    }

    common_names = sorted(set(left_files.keys()) & set(right_files.keys()))

    pairs = [(left_files[name], right_files[name]) for name in common_names]
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--left-calib", required=True)
    parser.add_argument("--right-calib", required=True)
    parser.add_argument("--output", default="stereo_calibration.npz")
    args = parser.parse_args()

    dictionary = make_dictionary()
    board = make_board(dictionary)

    K_left, D_left, image_size_left = load_calibration(args.left_calib)
    K_right, D_right, image_size_right = load_calibration(args.right_calib)

    if image_size_left != image_size_right:
        raise RuntimeError(
            f"Image sizes differ: left={image_size_left}, right={image_size_right}"
        )

    image_size = image_size_left

    pairs = paired_image_paths(args.left_dir, args.right_dir)

    if not pairs:
        raise RuntimeError("No matching left/right image pairs found.")

    all_object_points = []
    all_left_points = []
    all_right_points = []
    used_pairs = []

    print(f"Found {len(pairs)} candidate stereo pairs.")

    for left_path, right_path in pairs:
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            print(f"Skipping unreadable pair: {left_path}, {right_path}")
            continue

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        left_corners, left_ids = detect_charuco(left_gray, dictionary, board)
        right_corners, right_ids = detect_charuco(right_gray, dictionary, board)

        if left_corners is None or right_corners is None:
            print(f"Skipping pair: {os.path.basename(left_path)} - not enough corners")
            continue

        obj_pts, img_left, img_right, common_ids = common_charuco_points(
            left_corners,
            left_ids,
            right_corners,
            right_ids,
            board,
        )

        if obj_pts is None:
            print(f"Skipping pair: {os.path.basename(left_path)} - not enough common corners")
            continue

        all_object_points.append(obj_pts)
        all_left_points.append(img_left)
        all_right_points.append(img_right)
        used_pairs.append((left_path, right_path))

        print(
            f"Using pair {os.path.basename(left_path)}: "
            f"{len(common_ids)} common corners"
        )

    if len(all_object_points) < 15:
        raise RuntimeError(
            f"Only {len(all_object_points)} usable stereo pairs. "
            "Capture more varied pairs."
        )

    print()
    print(f"Running stereo calibration using {len(all_object_points)} stereo pairs...")
    print("Keeping individual camera intrinsics fixed.")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )

    flags = cv2.CALIB_FIX_INTRINSIC

    rms, K_left_out, D_left_out, K_right_out, D_right_out, R, T, E, F = cv2.stereoCalibrate(
        all_object_points,
        all_left_points,
        all_right_points,
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    baseline_m = float(np.linalg.norm(T))
    baseline_mm = baseline_m * 1000.0

    print()
    print("Stereo calibration complete.")
    print(f"Stereo RMS error: {rms:.4f} px")
    print()
    print("Rotation R, left camera to right camera:")
    print(R)
    print()
    print("Translation T, left camera to right camera, in metres:")
    print(T.ravel())
    print()
    print(f"Estimated baseline: {baseline_mm:.2f} mm")
    print()

    if 90.0 <= baseline_mm <= 110.0:
        print("Baseline looks reasonable for your 100 mm camera spacing.")
    else:
        print("WARNING: Baseline is not close to 100 mm.")
        print("Check board scale, saved pairs, camera order, and image quality.")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    np.savez(
        args.output,
        K_left=K_left,
        D_left=D_left,
        K_right=K_right,
        D_right=D_right,
        image_size=np.array(image_size, dtype=np.int32),
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=np.array(roi1),
        roi2=np.array(roi2),
        stereo_rms=np.array([rms], dtype=np.float64),
        baseline_m=np.array([baseline_m], dtype=np.float64),
        baseline_mm=np.array([baseline_mm], dtype=np.float64),
        squares_x=np.array([SQUARES_X], dtype=np.int32),
        squares_y=np.array([SQUARES_Y], dtype=np.int32),
        square_length_m=np.array([SQUARE_LENGTH_M], dtype=np.float64),
        marker_length_m=np.array([MARKER_LENGTH_M], dtype=np.float64),
        used_pairs=np.array(used_pairs),
    )

    print(f"Saved stereo calibration to: {args.output}")


if __name__ == "__main__":
    main()
