# Plan: freely configurable face detection & recognition models

Status: **in progress** on branch `claude/configurable-face-models`: steps 1 and 2 done, step 4 started. Independent of video
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

### 1. Make models configurable (no new architectures yet) — done

- `facialRecognition.detectionModelName` (empty = the detector of `modelName`), a flat field
  instead of a nested object, so no config migration is needed and existing clients keep working.
  The server sends both names in the `/predict` request, which already had separate `detection`
  and `recognition` entries.
- Both names accept the built-in packs or the full id of any Hugging Face repository
  (`owner/repository`). The server rejects anything else in `ConfigValidate`.
- ML: `ModelSource.HUGGINGFACE` for `owner/repository` names (face models only). They are downloaded
  from that repository at `main` and cached under `owner--repository`. The repository must use
  Immich's layout (`detection/model.onnx`, `recognition/model.onnx`); ARM NN/RKNN fall back to ONNX
  when missing.
- Web: free-text fields for both models instead of the fixed dropdown.
- **Moved to step 2:** a configurable detector `inputSize`. `FACE_DETECTION_SIZE` comes from the
  `immich-model` package, and the accelerated backends (RKNN, ARM NN) compile fixed shapes, so the
  input size belongs in the per-model descriptor.

### 2. Model descriptors instead of hard-coded preprocessing — done

- Optional `model.json` next to `detection/model.onnx` / `recognition/model.onnx`
  (`immich_ml/models/facial_recognition/spec.py`). Flat, snake_case, only the fields that differ
  from the defaults; the defaults are the contract of Immich's own models, so they need none.
  Unknown fields are rejected.
  - Detection: `family` (`scrfd` | `yolo`), `input_size` (multiple of the largest stride),
    `mean`/`std` (default per family), `channels` (`rgb` | `bgr`), `nms_threshold`, and for
    SCRFD `strides` and `anchors_per_cell`.
  - Recognition: `input_size` (the ArcFace template is scaled to it), `mean`, `std`, `channels`.
- The descriptor is read after the download and before the graph is built, so `input_size` also
  shapes the ONNX Runtime session. It replaces the fixed `FACE_DETECTION_SIZE`.
- Decoders: `decode_scrfd` (now parameterized) and a new `decode_yolo` for Ultralytics-style
  pose exports with 5 keypoints (YOLOv8-face, YOLO11-face). RetinaFace and YuNet are not added
  yet.
- `scripts/face_model_repo.py` packages ONNX files into the repository layout, writes the
  descriptor and can try the result on an image.
- Verified with real models: the plain (float input) InsightFace `buffalo_sc` pack run through
  this path with default settings gives the same boxes and embeddings (cosine 1.0) as the fused
  form Immich publishes, and also works at `input_size` 320 and 480.
- **Not verified yet:** a real YOLO face model (the unit tests use synthetic outputs; Hugging Face
  was not reachable from the development environment). Our letterbox pads at the bottom right with
  black, while Ultralytics centers with grey (114), which may cost a little accuracy.
- Not done: `embedding_size` in the descriptor (see step 3) and local directories mounted into the
  ML container (a local cache directory with the same layout already works).

### 3. Embedding size and switching models

- Embeddings from different recognition models are **not comparable**. Switching models must
  invalidate all ML faces: warn in the UI and offer "re-run face detection for all assets", as
  CLIP does on a model change.
- If `embedding_size` ≠ 512: recreate `face_search` with the new `vector(n)` size and rebuild the
  index, following the CLIP dimension-change handling in `SearchRepository` / `DatabaseRepository`.

### 4. Evaluation harness (to answer "how much better") — first version done

Done: `machine-learning/scripts/face_benchmark.py` runs models in the repository layout through the
real `FaceDetector`/`FaceRecognizer` classes:

- recognition: verification of labeled pairs (ROC AUC, best accuracy, equal error rate) on clean
  images and with the second image degraded (face 32/20 px, blur, motion blur, JPEG q10, low light)
- detection: faces found in group photos at full size and downscaled, against the faces at least
  two detectors agree on, plus hand counts
- speed per image/face on the current CPU

First results are in `plans/face-model-benchmark.md`. The public datasets below could not be
downloaded from the development environment (Hugging Face, Google Drive, figshare and Kaggle are
blocked there), so they are still open:

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
