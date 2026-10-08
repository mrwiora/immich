# Plan: freely configurable face detection & recognition models

Status: **in progress** on branch `claude/configurable-face-models` (step 1). Independent of video
frame sampling for face detection (`machineLearning.facialRecognition.videoFrameInterval`, branch
`claude/happy-hawking-kwd97p`), but most useful together with it.

## Goal

Find out how much better other face detection and recognition models are than the current
InsightFace packs. Measure it on public benchmarks **and** on a real Immich library, and keep the
model choice freely configurable so new models can be tried without code changes.

## Current state

| Part          | Where                                                                     | Limitation                                                                                |
| ------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Config        | `facialRecognition.modelName` in `server/src/dtos/config.dto.ts`          | One name selects the **pack** (detector + recognizer together)                            |
| Web UI        | `MachineLearningSettings.svelte`                                          | Hard-coded dropdown: `antelopev2`, `buffalo_l/m/s`                                        |
| ML allow-list | `_INSIGHTFACE_MODELS` in `machine-learning/immich_ml/models/constants.py` | Unknown names are rejected; source is always `ModelSource.INSIGHTFACE`                    |
| Detection     | `facial_recognition/detection.py` + `_ops.py`                             | SCRFD-specific output decoding (`decode_scrfd`), fixed square input `FACE_DETECTION_SIZE` |
| Recognition   | `facial_recognition/recognition.py` + `_ops.py`                           | ArcFace 5-point alignment to 112×112, normalization `(x - 127.5) / 127.5`, RGB            |
| Storage       | `face_search.embedding` is `vector(512)` with an HNSW index               | Models with another embedding size need a schema change                                   |
| Download      | `huggingface_hub.snapshot_download` from the `immich-app` HF org          | Only pre-exported ONNX models in Immich's own layout                                      |

## Candidates to evaluate

| Type        | Model                                                  | Expected gain                                                             | Notes                                                             |
| ----------- | ------------------------------------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Detection   | SCRFD-10G (current, `buffalo_l`)                       | Baseline                                                                  | Fast, good on WIDER FACE hard                                     |
| Detection   | SCRFD-10G at larger input (960/1280) or tiled          | More small faces (group photos, wide video shots)                         | Pure configuration once `inputSize` is configurable               |
| Detection   | RetinaFace-R50                                         | Slightly better recall                                                    | Slower; different decoder (anchors)                               |
| Detection   | YOLOv8/YOLO11-face (5 keypoints)                       | Comparable or better, fast on GPU                                         | Different decoder; check licensing (AGPL for Ultralytics weights) |
| Detection   | YuNet                                                  | Very fast on CPU                                                          | Weaker on hard faces; useful for low-end hardware                 |
| Recognition | ArcFace R50 / WebFace600K (current, `buffalo_l`)       | Baseline                                                                  |                                                                   |
| Recognition | ArcFace R100 / Glint360K (`antelopev2`)                | Small gain                                                                | ~2× slower, already supported                                     |
| Recognition | AdaFace IR-101 (WebFace4M/12M)                         | Clearly better on low-quality, blurry or side-on faces, i.e. video frames | Same 112×112 alignment; BGR input                                 |
| Recognition | EdgeFace / small models                                | Much faster                                                               | Less accurate                                                     |
| Quality     | AdaFace feature norm, CR-FIQA, or blur/size heuristics | Filters faces that harm clustering                                        | Most useful together with video sampling                          |

**Licensing:** InsightFace pretrained weights are for non-commercial research only. Most models
trained on MS1M or WebFace have similar dataset restrictions. Record each candidate's license
before shipping it.

## Steps

### 1. Make models configurable (no new architectures yet)

- Split the config into independent parts. Keep `modelName` and migrate it to both new names
  (system config migration):
  ```ts
  facialRecognition: {
    detection: { modelName: 'buffalo_l', inputSize: 640, minScore: 0.7 },
    recognition: { modelName: 'buffalo_l' },
    ...
  }
  ```
- Server: send the two model names separately in the `/predict` request. The request format
  already has separate `detection` and `recognition` entries
  (`server/src/repositories/machine-learning.repository.ts`).
- ML: let `FaceRecognizer.depends` work with a detector from a different pack. Read `inputSize`
  from the request options instead of `FACE_DETECTION_SIZE`.
- Web: replace the fixed dropdown with a free-text field that suggests known names. The CLIP model
  setting already allows any name.

### 2. Model descriptors instead of hard-coded preprocessing

- Ship a small `model.json` next to each ONNX file, for example:
  ```json
  {
    "embedding_size": 512,
    "family": "arcface",
    "input": { "channels": "rgb", "mean": 127.5, "size": 112, "std": 127.5 },
    "task": "facial-recognition",
    "type": "recognition"
  }
  ```
  For detectors: `family: scrfd | retinaface | yolo-face | yunet`, `strides`, `anchors`,
  `keypoints: 5`.
- Pick decoders by `family` (`decode_scrfd`, `decode_retinaface`, `decode_yolo`). Recognition
  families share `align_face` and differ only in normalization and color order.
- Allow models from any Hugging Face repo, or a local directory mounted into the ML container, as
  long as a `model.json` is present. Fall back to today's built-in list for the existing names.

### 3. Embedding size and switching models

- Embeddings from different recognition models are **not comparable**. Switching models must
  invalidate all ML faces: warn in the UI and offer "re-run face detection for all assets", as
  CLIP does on a model change.
- If `embedding_size` ≠ 512: recreate `face_search` with the new `vector(n)` size and rebuild the
  index, following the CLIP dimension-change handling in `SearchRepository` / `DatabaseRepository`.

### 4. Evaluation harness (to answer "how much better")

Do this offline so the live library isn't touched.

- `machine-learning/scripts/face_benchmark.py`:
  - **Detection:** WIDER FACE val (AP easy/medium/hard), plus recall at the configured `minScore`.
  - **Recognition:** LFW, CFP-FP, AgeDB-30 (accuracy), IJB-C (TAR@FAR=1e-4).
  - **Library ground truth:** export faces that users assigned or confirmed to people from an Immich
    database (`asset_face` + `person`, with bounding boxes and preview paths). Run each
    detector/recognizer pair and replay the clustering (same `maxDistance` / `minFaces` logic as
    `PersonService.handleRecognizeFaces`). Report pairwise precision/recall, BCubed F1, number of
    people vs. ground truth, and the share of unassigned faces.
  - **Video subset:** sample frames with `getVideoFrameTimestamps` and measure detections per
    video and duplicate faces per person after `groupVideoFaces`.
  - **Cost:** latency per image (CPU, CUDA, OpenVINO), peak memory, model size.
- Output: a table per model combination, so we can choose the best default and good options for
  "small hardware" and "max quality".

### 5. Optional: compare models side by side in a live library

- Store embeddings per model (an extra `face_search_eval(faceId, model, embedding)` table) behind a
  hidden admin flag. A benchmark job can then compute the metrics above against the user's own
  people without changing what users see.

## Open questions

- Should detection and recognition model names also be allowed per user, or only server-wide?
  (Probably server-wide: embeddings must be comparable across a cluster group.)
- Should the quality filter be its own setting (`minQuality`) or part of `minScore`?
- Which candidates do we host on the Immich HF org vs. load from user-provided locations?
