"""
Host-side geometry for the face models.

The SCRFD decode, NMS and Umeyama similarity are ports of insightface
(Apache-2.0, https://github.com/deepinsight/insightface)
"""

from functools import lru_cache

import cv2
import numpy as np
from immich_model.constants import FACE_DETECTION_SIZE as DET_SIZE
from numpy.typing import NDArray

from immich_ml.models.transforms import ensure_dims

ALIGNED_SIZE = 112

# the FPN levels for which the fused detector emits a (scores, boxes, kps) triple,
# and the anchors each feature-map cell carries, laid out anchor-major
DET_STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2

# canonical ArcFace 5-point template for a 112x112 crop
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@lru_cache(maxsize=16)
def _anchor_centers(size: int, stride: int, anchors: int = ANCHORS_PER_CELL) -> NDArray[np.float32]:
    ys, xs = np.mgrid[: size // stride, : size // stride]
    centers = np.stack([xs, ys], axis=-1, dtype=np.float32).reshape(-1, 2) * stride
    return np.repeat(centers, anchors, axis=0)


def decode_scrfd(
    heads: list[NDArray[np.float32]],
    size: int = DET_SIZE,
    strides: tuple[int, ...] = DET_STRIDES,
    anchors: int = ANCHORS_PER_CELL,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    heads = [ensure_dims(head, 4) for head in heads]
    scores, boxes, kps = [], [], []
    for level, stride in enumerate(strides):
        centers = _anchor_centers(size, stride, anchors)
        distance = heads[level + len(strides)][0] * stride
        offsets = heads[level + 2 * len(strides)][0] * stride
        scores.append(heads[level][0].squeeze(-1))
        boxes.append(np.concatenate([centers[None] - distance[:, :, :2], centers[None] + distance[:, :, 2:]], axis=2))
        kps.append(np.tile(centers, offsets.shape[2] // 2) + offsets)

    return np.concatenate(scores, axis=1), np.concatenate(boxes, axis=1), np.concatenate(kps, axis=1)


def decode_yolo(
    heads: list[NDArray[np.float32]],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Decodes a YOLO pose-style face model (e.g. YOLOv8-face, YOLO11-face) in the Ultralytics export layout:
    one output of [cx, cy, w, h, score, 5 x (x, y, visibility)] per candidate, channels first or last.
    """
    output = ensure_dims(heads[0], 3)[0]
    if output.shape[0] < output.shape[1]:  # channels first, as there are far more candidates than channels
        output = output.T
    if output.shape[1] != 20:
        raise ValueError(f"Expected 20 values per YOLO face candidate, got {output.shape[1]}")

    centers, sizes = output[:, :2], output[:, 2:4] / 2
    boxes = np.concatenate([centers - sizes, centers + sizes], axis=1)
    kps = output[:, 5:].reshape(-1, 5, 3)[:, :, :2].reshape(-1, 10)
    return output[None, :, 4], boxes[None], kps[None]


def nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], threshold: float = 0.4) -> NDArray[np.intp]:
    wh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
    keep = cv2.dnn.NMSBoxes(wh.tolist(), scores.tolist(), 0.0, threshold)  # NMSBoxes treats the inputs as Sequences
    return np.asarray(keep, dtype=np.intp).reshape(-1)


def umeyama(src: NDArray[np.float32], dst: NDArray[np.float32]) -> NDArray[np.float32]:
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.diag([1.0, d])
    rotation = u @ diag @ vt
    scale = np.trace(np.diag(s) @ diag) / (src_c**2).sum() * len(src)
    translation = dst_mean - scale * rotation @ src_mean
    return np.hstack([scale * rotation, translation[:, None]], dtype=np.float32)


def align_face(image: NDArray[np.uint8], kps: NDArray[np.float32], crop: NDArray[np.uint8]) -> None:
    """Aligns to the ArcFace template, scaled to the size of the crop the recognizer expects."""
    size = crop.shape[0]
    cv2.warpAffine(image, umeyama(kps, ARCFACE_DST * (size / ALIGNED_SIZE)), (size, size), dst=crop)
