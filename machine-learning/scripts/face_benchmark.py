"""
Compares face models in Immich's repository layout (see face_model_repo.py) on your own images.

Recognition: verification of labeled image pairs. The faces are found once with a reference detector, so only the
recognizers differ, and are degraded on the second image of each pair (low resolution, blur, JPEG, low light) the way
video frames and small faces in group photos are. Reports ROC AUC, the best accuracy and the equal error rate.

Detection: the faces each detector finds in group photos at full size and downscaled. As group photos rarely come
with annotations, the reference is every face at least two detectors agree on at full size, plus optional hand counts.

    uv run python scripts/face_benchmark.py \\
        --model buffalo_l=./repos/buffalo_l --model adaface_ir101=./repos/adaface_ir101 \\
        --pairs pairs.csv --images ./faces --groups ./group-photos --output report.md

`pairs.csv` has the columns file_x, file_y and decision (Yes/No); `--groups` may hold a `counts.json` of hand counts.
"""

import argparse
import csv
import io
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from immich_ml.models.facial_recognition.detection import FaceDetector
from immich_ml.models.facial_recognition.recognition import FaceRecognizer
from immich_ml.schemas import FaceDetectionOptions, FaceDetectionOutput, FaceRecognitionOptions, ModelFormat

Degradation = Callable[[Image.Image, float], Image.Image]


def low_resolution(pixels: int) -> Degradation:
    """Shrinks the image until the face is `pixels` wide, then back, keeping the landmarks where they were."""

    def degrade(image: Image.Image, face_width: float) -> Image.Image:
        factor = min(1.0, pixels / face_width)
        small = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))))
        return small.resize(image.size, Image.Resampling.BILINEAR)

    return degrade


def blur(image: Image.Image, face_width: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=face_width / 40))


def motion_blur(image: Image.Image, face_width: float) -> Image.Image:
    length = max(3, round(face_width / 12))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2] = 1 / length
    return Image.fromarray(cv2.filter2D(np.asarray(image), -1, kernel))


def jpeg(image: Image.Image, face_width: float) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=10)
    return Image.open(buffer).convert("RGB")


def low_light(image: Image.Image, face_width: float) -> Image.Image:
    dark = (np.asarray(image, dtype=np.float32) / 255) ** 2.5 * 255
    noisy = dark + np.random.default_rng(0).normal(0, 6, dark.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def video_frame(image: Image.Image, face_width: float) -> Image.Image:
    """A small face in a compressed, moving video frame."""
    return jpeg(motion_blur(low_resolution(24)(image, face_width), 24), face_width)


DEGRADATIONS: dict[str, Degradation | None] = {
    "original": None,
    "face 32px": low_resolution(32),
    "face 20px": low_resolution(20),
    "face 14px": low_resolution(14),
    "blur": blur,
    "motion blur": motion_blur,
    "jpeg q10": jpeg,
    "low light": low_light,
    "video frame (24px, motion blur, JPEG)": video_frame,
}

# Immich's default maximum cosine distance for two faces to be the same person
IMMICH_MAX_DISTANCE = 0.5


@dataclass
class Model:
    name: str
    path: Path
    detector: FaceDetector | None = None
    recognizer: FaceRecognizer | None = None
    timings: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def load(cls, spec: str) -> "Model":
        name, _, path = spec.partition("=")
        model = cls(name, Path(path))
        if (model.path / "detection" / "model.onnx").is_file():
            model.detector = FaceDetector("local/model", cache_dir=model.path, model_format=ModelFormat.ONNX)
        if (model.path / "recognition" / "model.onnx").is_file():
            model.recognizer = FaceRecognizer("local/model", cache_dir=model.path, model_format=ModelFormat.ONNX)
        return model

    def timed[R](self, kind: str, func: Callable[[], R]) -> R:
        start = time.perf_counter()
        result = func()
        self.timings.setdefault(kind, []).append(time.perf_counter() - start)
        return result

    def detect(self, image: Image.Image, min_score: float) -> FaceDetectionOutput:
        assert self.detector is not None
        detector = self.detector
        faces: FaceDetectionOutput = self.timed(
            "detection", lambda: detector.predict(image, options=FaceDetectionOptions(min_score))
        )
        return faces

    def embed(self, image: Image.Image, face: FaceDetectionOutput) -> NDArray[np.float32]:
        assert self.recognizer is not None
        recognizer = self.recognizer
        output = self.timed("recognition", lambda: recognizer.predict(image, face, options=FaceRecognitionOptions()))
        embedding = np.asarray(json.loads(output[0]["embedding"]), dtype=np.float32)
        normalized: NDArray[np.float32] = (embedding / np.linalg.norm(embedding)).astype(np.float32)
        return normalized


def largest_face(faces: FaceDetectionOutput) -> FaceDetectionOutput | None:
    if len(faces["boxes"]) == 0:
        return None
    boxes = faces["boxes"]
    i = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    return {"boxes": boxes[i : i + 1], "scores": faces["scores"][i : i + 1], "landmarks": faces["landmarks"][i : i + 1]}


def verification_metrics(similarities: NDArray[np.float32], same: NDArray[np.bool_]) -> dict[str, float]:
    genuine, impostor = similarities[same], similarities[~same]
    # how far apart the two similarity distributions are, in standard deviations; unlike accuracy it keeps
    # telling models apart when all of them separate a small set perfectly
    dprime = float((genuine.mean() - impostor.mean()) / np.sqrt((genuine.var() + impostor.var()) / 2))
    threshold = np.quantile(impostor, 0.999)
    return {
        "dprime": dprime,
        "tar": float((genuine > threshold).mean()),
        "immich_tar": float((genuine >= 1 - IMMICH_MAX_DISTANCE).mean()),
        "immich_far": float((impostor >= 1 - IMMICH_MAX_DISTANCE).mean()),
    }


def benchmark_recognition(
    models: list[Model], reference: Model, pairs: Path, images: Path, min_score: float
) -> Iterator[str]:
    rows = list(csv.DictReader(pairs.open()))
    names = sorted({row["file_x"] for row in rows} | {row["file_y"] for row in rows})
    loaded = {name: Image.open(images / name).convert("RGB") for name in names}
    faces = {name: largest_face(reference.detect(image, min_score)) for name, image in loaded.items()}
    missing = sorted(name for name, face in faces.items() if face is None)
    rows = [row for row in rows if faces[row["file_x"]] is not None and faces[row["file_y"]] is not None]
    same = np.array([row["decision"].lower() == "yes" for row in rows])

    recognizers = [model for model in models if model.recognizer is not None]
    yield f"## Recognition\n\n{len(rows)} pairs ({same.sum()} same person) of {len(names)} images, faces found by "
    yield f"`{reference.name}`" + (f"; no face in {', '.join(missing)}" if missing else "") + ".\n\n"
    yield "Each cell: d′ (separation of same/different person similarities, higher is better) · same-person pairs "
    yield "matched at a 0.1% false match rate · at Immich's default threshold (cosine distance ≤ "
    yield f"{IMMICH_MAX_DISTANCE}): same-person pairs matched / different-person pairs wrongly matched.\n\n"
    yield "| Degradation of the 2nd image | " + " | ".join(model.name for model in recognizers) + " |\n"
    yield "|---|" + "---|" * len(recognizers) + "\n"
    results: dict[str, list[str]] = {label: [] for label in DEGRADATIONS}
    for model in recognizers:
        clean = {name: model.embed(loaded[name], face) for name, face in faces.items() if face is not None}
        for label, degrade in DEGRADATIONS.items():
            degraded = clean
            if degrade is not None:
                degraded = {
                    name: model.embed(degrade(loaded[name], float(face["boxes"][0, 2] - face["boxes"][0, 0])), face)
                    for name, face in faces.items()
                    if face is not None
                }
            similarities = [float(clean[row["file_x"]] @ degraded[row["file_y"]]) for row in rows]
            metrics = verification_metrics(np.asarray(similarities, dtype=np.float32), same)
            immich = f"{metrics['immich_tar']:.0%} / {metrics['immich_far']:.1%}"
            results[label].append(f"{metrics['dprime']:.2f} · {metrics['tar']:.0%} · {immich}")
    for label, cells in results.items():
        yield f"| {label} | " + " | ".join(cells) + " |\n"
    yield "\n"


def iou(box: NDArray[np.float32], boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    areas = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    result: NDArray[np.float32] = intersection / (areas - intersection)
    return result


def matched(found: NDArray[np.float32], reference: NDArray[np.float32], threshold: float = 0.4) -> int:
    """Greedy one-to-one matching of found boxes to reference boxes."""
    taken = np.zeros(len(reference), dtype=bool)
    for box in found:
        if len(reference) == 0:
            break
        overlaps = np.where(taken, 0, iou(box, reference))
        best = int(np.argmax(overlaps))
        if overlaps[best] >= threshold:
            taken[best] = True
    return int(taken.sum())


def consensus(boxes: list[NDArray[np.float32]], threshold: float = 0.4) -> NDArray[np.float32]:
    """The boxes at least two detectors found, averaged."""
    agreed: list[NDArray[np.float32]] = []
    for i, own in enumerate(boxes):
        for box in own:
            closest = [other[np.argmax(iou(box, other))] for other in boxes[:i] + boxes[i + 1 :] if len(other)]
            confirming = [other for other in closest if iou(box, other[None])[0] >= threshold]
            known = np.asarray(agreed, dtype=np.float32).reshape(-1, 4)
            if confirming and not (iou(box, known) >= threshold).any():
                agreed.append(np.mean([box, *confirming], axis=0, dtype=np.float32))
    return np.asarray(agreed, dtype=np.float32).reshape(-1, 4)


def benchmark_detection(
    models: list[Model], groups: Path, min_score: float, scales: list[float], voters: list[str] | None = None
) -> Iterator[str]:
    counts_file = groups / "counts.json"
    counts: dict[str, int] = json.loads(counts_file.read_text()) if counts_file.is_file() else {}
    detectors = [model for model in models if model.detector is not None]
    paths = sorted(path for path in groups.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

    yield f"## Detection\n\nFaces found (minimum score {min_score}) / faces of the reference found. The reference is "
    yield "every face at least two detectors agree on at full size; hand counts where available.\n\n"
    header = " | ".join(f"{model.name} @ {scale:g}" for scale in scales for model in detectors)
    yield f"| Image | Reference | Hand count | {header} |\n|---|---|---|" + "---|" * len(scales) * len(detectors) + "\n"
    totals = np.zeros((len(scales), len(detectors)), dtype=int)
    reference_total = 0
    for path in paths:
        image = Image.open(path).convert("RGB")
        found: dict[tuple[float, str], NDArray[np.float32]] = {}
        for scale in scales:
            scaled = image.resize((round(image.width * scale), round(image.height * scale)))
            for model in detectors:
                found[scale, model.name] = model.detect(scaled, min_score)["boxes"] / scale
        reference = consensus([found[1.0, model.name] for model in detectors if not voters or model.name in voters])
        reference_total += len(reference)
        cells = []
        for s, scale in enumerate(scales):
            for d, model in enumerate(detectors):
                hits = matched(found[scale, model.name], reference)
                totals[s, d] += hits
                cells.append(f"{len(found[scale, model.name])} / {hits}")
        yield f"| {path.name} | {len(reference)} | {counts.get(path.name, '')} | " + " | ".join(cells) + " |\n"
    recall = " | ".join(
        f"{totals[s, d] / max(reference_total, 1):.1%}" for s in range(len(scales)) for d in range(len(detectors))
    )
    yield f"| **share of the reference found** | {reference_total} | | {recall} |\n\n"


def speed(models: list[Model]) -> Iterator[str]:
    yield "## Speed (CPU, this machine)\n\n| Model | Detection per image | Recognition per face |\n|---|---|---|\n"
    for model in models:
        cells = [
            f"{np.median(model.timings[kind]) * 1000:.0f} ms" if kind in model.timings else ""
            for kind in ("detection", "recognition")
        ]
        yield f"| {model.name} | " + " | ".join(cells) + " |\n"
    yield "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="NAME=DIRECTORY, repeatable")
    parser.add_argument("--pairs", type=Path, help="CSV of labeled image pairs (file_x, file_y, decision)")
    parser.add_argument("--images", type=Path, help="directory of the images in --pairs")
    parser.add_argument("--reference-detector", help="model whose detector finds the faces for recognition")
    parser.add_argument("--groups", type=Path, help="directory of group photos for detection")
    parser.add_argument("--scales", default="1,0.5,0.25", help="sizes to detect the group photos at")
    parser.add_argument(
        "--reference-models",
        help="comma-separated detectors whose agreement makes the detection reference (default: all); "
        "leave out variants of the same network, as they always agree",
    )
    parser.add_argument("--min-score", type=float, default=0.7, help="minimum detection score, as in Immich")
    parser.add_argument("--output", type=Path, help="write the report here as well")
    args = parser.parse_args()

    models = [Model.load(spec) for spec in args.model]
    report = ["# Face model benchmark\n\n"]
    if args.pairs is not None:
        detectors = [model for model in models if model.detector is not None]
        reference = next((model for model in detectors if model.name == args.reference_detector), detectors[0])
        report += benchmark_recognition(models, reference, args.pairs, args.images, args.min_score)
    if args.groups is not None:
        scales = [float(scale) for scale in args.scales.split(",")]
        voters = args.reference_models.split(",") if args.reference_models else None
        report += benchmark_detection(models, args.groups, args.min_score, scales, voters)
    report += speed(models)

    text = "".join(report)
    print(text)
    if args.output is not None:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
