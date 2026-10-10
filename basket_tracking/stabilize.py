"""Camera-motion compensation for handheld phone video.

For every frame we estimate a homography H that maps frame pixels into the pixels of a
fixed reference frame. A phone held from one spot mostly rotates, so a single homography
per frame describes the whole scene well. Positions expressed in reference-frame pixels are
then comparable across the whole clip, which is what tracking and heat maps need.
"""

import cv2
import numpy as np

MIN_INLIERS = 40


class Stabilizer:
    def __init__(self, reference_frame: np.ndarray, n_features=4000):
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ref_kp, self.ref_des = self._features(reference_frame, mask=None)
        self.prev = None  # (keypoints, descriptors, H_prev_to_ref)

    def _features(self, frame, mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, mask)

    def _match(self, kp_a, des_a, kp_b, des_b):
        """Homography mapping points of a into b, plus inlier count."""
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return None, 0
        pairs = self.matcher.knnMatch(des_a, des_b, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            return None, 0
        src = np.float32([kp_a[m.queryIdx].pt for m in good])
        dst = np.float32([kp_b[m.trainIdx].pt for m in good])
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0
        return H, int(inliers.sum())

    def update(self, frame: np.ndarray, player_boxes: np.ndarray) -> np.ndarray:
        """Return H mapping `frame` pixels to reference pixels. Players are masked out
        because they move independently of the camera."""
        mask = np.full(frame.shape[:2], 255, np.uint8)
        for x1, y1, x2, y2 in player_boxes.astype(int):
            mask[max(y1, 0):y2, max(x1, 0):x2] = 0
        kp, des = self._features(frame, mask)

        # Match straight to the reference (no drift); fall back to chaining via the
        # previous frame when the view has moved too far from the reference.
        H, inliers = self._match(kp, des, self.ref_kp, self.ref_des)
        if inliers < MIN_INLIERS and self.prev is not None:
            prev_kp, prev_des, H_prev = self.prev
            H_step, step_inliers = self._match(kp, des, prev_kp, prev_des)
            H = H_prev @ H_step if step_inliers >= MIN_INLIERS else H_prev
        if H is None:
            H = np.eye(3)
        self.prev = (kp, des, H)
        return H


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def transform_boxes(H: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Map xyxy boxes through H, returning the axis-aligned box around the warped corners."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.c_[x1, y1], np.c_[x2, y1], np.c_[x2, y2], np.c_[x1, y2]], axis=1)
    warped = transform_points(H, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.c_[warped.min(axis=1), warped.max(axis=1)]
