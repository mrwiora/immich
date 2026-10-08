"""
Packages face detection and/or recognition ONNX models into the layout Immich loads from a Hugging Face repository,
with a `model.json` descriptor for anything that differs from the defaults, and optionally tries them on an image.

    uv run python scripts/face_model_repo.py ./my-repo \\
        --detection yolo11n-face.onnx --detection-spec '{"family": "yolo"}' \\
        --recognition adaface_ir101.onnx --recognition-spec '{"channels": "bgr"}' \\
        --image group-photo.jpg
    hf upload <owner>/<repository> ./my-repo

Then set the facial recognition and/or face detection model in Immich to `<owner>/<repository>`.
"""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from PIL import Image
from pydantic import TypeAdapter

from immich_ml.models.facial_recognition.detection import FaceDetector
from immich_ml.models.facial_recognition.recognition import FaceRecognizer
from immich_ml.models.facial_recognition.spec import DetectorSpec, RecognizerSpec
from immich_ml.schemas import FaceDetectionOptions, FaceRecognitionOptions, ModelFormat


def package(model: Path, spec: DetectorSpec | RecognizerSpec, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, model_dir / "model.onnx")
    descriptor = {key: value for key, value in asdict(spec).items() if value != asdict(type(spec)())[key]}
    if descriptor:
        (model_dir / "model.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    print(f"Wrote {model_dir / 'model.onnx'}" + (f" with {descriptor}" if descriptor else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="directory to write the repository to")
    parser.add_argument("--detection", type=Path, help="face detection ONNX model")
    parser.add_argument("--detection-spec", default="{}", help="JSON descriptor of the detection model")
    parser.add_argument("--recognition", type=Path, help="face recognition ONNX model")
    parser.add_argument("--recognition-spec", default="{}", help="JSON descriptor of the recognition model")
    parser.add_argument("--image", type=Path, help="image to detect and recognize faces in with the result")
    parser.add_argument("--min-score", type=float, default=0.7, help="minimum detection score for --image")
    args = parser.parse_args()

    if args.detection is None and args.recognition is None:
        parser.error("at least one of --detection and --recognition is required")
    if args.detection is not None:
        detector_spec = TypeAdapter(DetectorSpec).validate_json(args.detection_spec)
        package(args.detection, detector_spec, args.output / "detection")
    if args.recognition is not None:
        recognizer_spec = TypeAdapter(RecognizerSpec).validate_json(args.recognition_spec)
        package(args.recognition, recognizer_spec, args.output / "recognition")

    if args.image is None:
        return
    if args.detection is None or args.recognition is None:
        parser.error("--image needs both --detection and --recognition")

    image = Image.open(args.image).convert("RGB")
    detector = FaceDetector("local/model", cache_dir=args.output, model_format=ModelFormat.ONNX)
    recognizer = FaceRecognizer("local/model", cache_dir=args.output, model_format=ModelFormat.ONNX)
    faces = detector.predict(image, options=FaceDetectionOptions(min_score=args.min_score))
    embeddings = recognizer.predict(image, faces, options=FaceRecognitionOptions())
    print(f"Found {len(embeddings)} faces in {args.image}:")
    for face in embeddings:
        box = ", ".join(f"{value:.0f}" for value in face["boundingBox"].values())
        dims = len(json.loads(face["embedding"]))
        print(f"  box ({box}), score {face['score']:.2f}, {dims}-dimensional embedding")


if __name__ == "__main__":
    main()
