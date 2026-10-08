from typing import Any

import numpy as np
from numpy.typing import NDArray

from immich_ml.models.base import InferenceModel
from immich_ml.models.transforms import decode_pil, letterbox, normalize, widen
from immich_ml.schemas import (
    FaceDetectionOptions,
    FaceDetectionOutput,
    ModelSession,
    ModelSource,
    ModelTask,
    ModelType,
    Shape,
)
from immich_ml.sessions.policy import ShapePolicy

from ._ops import decode_scrfd, decode_yolo, nms
from .spec import DetectorSpec, read_spec


class FaceDetector(InferenceModel[FaceDetectionOptions]):
    depends = []
    identity = (ModelType.DETECTION, ModelTask.FACIAL_RECOGNITION)
    sources = (ModelSource.INSIGHTFACE, ModelSource.HUGGINGFACE)

    def __init__(self, model_name: str, **model_kwargs: Any) -> None:
        super().__init__(model_name, **model_kwargs)
        self.configure(DetectorSpec())

    def configure(self, spec: DetectorSpec) -> None:
        self.spec = spec
        self.shape_policy = ShapePolicy(dims=(Shape(batch=1, height=spec.input_size, width=spec.input_size),))

    def _load(self) -> ModelSession:
        # the descriptor arrives with the model, so the graph is only shaped once it is downloaded
        self.configure(read_spec(self.model_dir, DetectorSpec))
        return super()._load()

    def _predict(self, inputs: NDArray[np.uint8] | bytes, options: FaceDetectionOptions) -> FaceDetectionOutput:
        spec = self.spec
        session = self.session.for_shape(self.shape_policy.dims[0])
        canvas, scale = letterbox(decode_pil(inputs), spec.input_size)
        if spec.channels == "bgr":
            canvas = np.ascontiguousarray(canvas[..., ::-1])
        blob: NDArray[np.float32] | NDArray[np.uint8] = (
            canvas[None]
            if session.normalizes_input
            else normalize(canvas.astype(np.float32), *spec.normalization).transpose(2, 0, 1)[None]
        )

        heads = [widen(head) for head in session.run(None, {session.get_inputs()[0].name: blob})]
        if spec.family == "yolo":
            scores, boxes, kps = decode_yolo(heads)
        else:
            scores, boxes, kps = decode_scrfd(heads, spec.input_size, spec.strides, spec.anchors_per_cell)

        candidates = scores >= options.min_score
        scores, boxes, kps = scores[candidates], boxes[candidates] / scale, kps[candidates] / scale
        keep = nms(boxes, scores, spec.nms_threshold)

        return {
            "boxes": boxes[keep].round(),
            "scores": scores[keep],
            "landmarks": kps[keep].reshape(-1, 5, 2),
        }
