"""
Optional `model.json` next to a face model, describing what the model expects and emits.

Immich's own models need none: the defaults are their contract. A model from any other repository brings one
when it differs, e.g. an AdaFace recognizer that reads BGR or a YOLO detector with its own decoding.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from immich_model.constants import FACE_DETECTION_SIZE
from pydantic import ConfigDict, TypeAdapter, with_config

from immich_ml.config import log

STRICT = ConfigDict(extra="forbid")

Channels = Literal["rgb", "bgr"]

# what an unfused graph (float input) is normalized with when the descriptor does not say: (mean, std)
_DETECTOR_NORMALIZATION = {"scrfd": (127.5, 128.0), "yolo": (0.0, 255.0)}


@with_config(STRICT)
@dataclass(frozen=True)
class DetectorSpec:
    family: Literal["scrfd", "yolo"] = "scrfd"
    input_size: int = FACE_DETECTION_SIZE
    mean: float | None = None
    std: float | None = None
    channels: Channels = "rgb"
    nms_threshold: float = 0.4
    # SCRFD: the FPN levels emitting a (scores, boxes, kps) triple, and the anchors each feature-map cell carries
    strides: tuple[int, ...] = (8, 16, 32)
    anchors_per_cell: int = 2

    def __post_init__(self) -> None:
        # both families downsample by up to the largest stride, which a canvas of another size leaves misaligned
        if self.input_size % max(self.strides):
            raise ValueError(f"input_size must be a multiple of {max(self.strides)}, got {self.input_size}")

    @property
    def normalization(self) -> tuple[float, float]:
        mean, std = _DETECTOR_NORMALIZATION[self.family]
        return (mean if self.mean is None else self.mean, std if self.std is None else self.std)


@with_config(STRICT)
@dataclass(frozen=True)
class RecognizerSpec:
    input_size: int = 112
    mean: float = 127.5
    std: float = 127.5
    channels: Channels = "rgb"


def read_spec[S: (DetectorSpec, RecognizerSpec)](model_dir: Path, spec: type[S]) -> S:
    path = model_dir / "model.json"
    if not path.is_file():
        return spec()
    loaded = TypeAdapter(spec).validate_json(path.read_bytes())
    log.info(f"Using {path}: {loaded}")
    return loaded
