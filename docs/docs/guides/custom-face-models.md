# Custom Face Models

## Purpose

Immich detects faces with one model and recognizes them with another. Besides the built-in InsightFace models (`buffalo_l`, `buffalo_m`, `buffalo_s`, `antelopev2`), both can be any compatible ONNX model. This guide explains which models work, how to try one and how to compare it with the default before you switch your library to it.

:::caution
Switching the **facial recognition** model makes all existing face embeddings incomparable with new ones: you need to re-run face detection for all assets afterwards. Switching only the **face detection** model affects new face detection jobs. Try models on a test instance or with the benchmark below first.
:::

:::info Licenses
Most pretrained face models are licensed for non-commercial research only, including Immich's built-in InsightFace models, and many are trained on datasets with similar terms. Check the license of a model and its training data before you use it.
:::

## Prerequisites

### Models that work

**Face detection**, as ONNX with a square input (fixed or dynamic size):

- **SCRFD** (InsightFace) with keypoints: 9 outputs, scores, boxes and 5 keypoints for each of the strides 8, 16 and 32. This is the default.
- **YOLO pose-style face models with 5 keypoints**, such as YOLOv8-face and YOLO11-face, in either export layout:
  - Ultralytics: one output with `[cx, cy, w, h, score, 5 × (x, y, visibility)]` per candidate
  - raw (yolov8-face project): one feature map with 80 channels per stride

**Face recognition**, as ONNX:

- takes a face aligned to the 5-point ArcFace template (112 × 112 by default), as float NCHW input
- outputs one embedding of **512** values per face

Both may instead take the raw 8-bit image (NHWC), as Immich's own models do; then `mean` and `std` are not used.

### Models that do not work (yet)

- detectors with other outputs: RetinaFace, YOLOv5-face, YuNet, CenterFace, MediaPipe BlazeFace
- detectors without the 5 face keypoints, as recognition depends on them for alignment
- recognizers with an embedding size other than 512 (e.g. 128) or another alignment (e.g. SphereFace 112 × 96)

### Tools

The scripts below are part of the machine learning service. Run them from a checkout of the Immich repository:

```bash
cd machine-learning
uv sync --extra cpu
```

## Models that were tested

All of these were run through Immich's own detection and recognition code. The ONNX files are exports by [yakhyo](https://github.com/yakhyo) unless they come from InsightFace.

| Model                               | Type        | ONNX file                                                                                                              | `model.json`           | Result compared to `buffalo_l`                                                                                                 |
| ----------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| SCRFD-10G (`buffalo_l`)             | Detection   | built in                                                                                                               | –                      | Baseline. Best overall at the default 640 px input.                                                                            |
| SCRFD-10G at 1280 px                | Detection   | `det_10g.onnx` from the InsightFace `buffalo_l` pack                                                                   | `{"input_size": 1280}` | ~3× more small faces in crowds, ~4× slower, misses very large close-up faces.                                                  |
| SCRFD-500M (`buffalo_s`)            | Detection   | built in                                                                                                               | –                      | ~5× faster, finds fewer small faces.                                                                                           |
| YOLOv8n-face                        | Detection   | [yolov8n-face.onnx](https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx)  | `{"family": "yolo"}`   | Better with partly visible faces, worse in crowds; needs a minimum score of ~0.5.                                              |
| ArcFace R50 (`buffalo_l`)           | Recognition | built in                                                                                                               | –                      | Baseline.                                                                                                                      |
| ArcFace R100 (`antelopev2`)         | Recognition | built in                                                                                                               | –                      | **Best** in every condition, ~1.8× slower.                                                                                     |
| ArcFace MobileFaceNet (`buffalo_s`) | Recognition | built in                                                                                                               | –                      | Clearly worse, ~4.5× faster.                                                                                                   |
| AdaFace IR-101 (WebFace12M)         | Recognition | [adaface_ir_101.onnx](https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_101.onnx)            | `{"channels": "bgr"}`  | Better than `buffalo_l`, below `antelopev2`, ~1.8× slower.                                                                     |
| AdaFace IR-50 (WebFace4M)           | Recognition | [adaface_ir_50.onnx](https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_50.onnx)              | `{"channels": "bgr"}`  | Slightly worse, same speed.                                                                                                    |
| EdgeFace-Base                       | Recognition | [edgeface_base.onnx](https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_base.onnx)             | –                      | Slightly worse, ~4× faster.                                                                                                    |
| EdgeFace-S (γ=0.5)                  | Recognition | [edgeface_s_gamma_05.onnx](https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_s_gamma_05.onnx) | –                      | Worse, ~7× faster; similar to `buffalo_s` but more robust with small faces and faster, so a good choice for low-power servers. |

The comparison used 61 photos of 13 people with simulated small, blurred, compressed and dark faces, and 8 group photos with up to several hundred faces. It is a small test; run the benchmark on your own photos before you decide.

## Step 1: Find out what the model expects

Look at the inference code that comes with the ONNX export and note:

- **color order**: does it convert the image to RGB, or keep OpenCV's BGR? (AdaFace uses BGR, most others RGB.)
- **normalization**: `(pixel - mean) / std`; common values are `127.5`/`127.5` (recognition), `127.5`/`128` (SCRFD) and `0`/`255` (YOLO)
- **input size**: 112 for most recognizers, 640 for most detectors
- for detectors, the **output layout** (see above)

Then write the differences from the defaults as a [`model.json`](/features/facial-recognition#custom-models). Getting the color order wrong does not fail, it just makes the model noticeably worse.

## Step 2: Package the model and try it on a photo

```bash
uv run python scripts/face_model_repo.py ./my-model \
  --detection yolov8n-face.onnx --detection-spec '{"family": "yolo"}' \
  --recognition adaface_ir_101.onnx --recognition-spec '{"channels": "bgr"}' \
  --image group-photo.jpg --min-score 0.5
```

This writes `detection/` and `recognition/` with the `model.onnx` and `model.json` files and prints the faces found in the photo. You can package only one of the two models; you can then combine it with a built-in model in Immich.

To package a built-in model for a comparison, take the ONNX files from an [InsightFace pack](https://github.com/deepinsight/insightface/tree/master/model_zoo), e.g. `det_10g.onnx` and `w600k_r50.onnx` from `buffalo_l.zip`.

## Step 3: Compare it with the default

`scripts/face_benchmark.py` compares packaged models with each other.

For **recognition** it needs photos of people you know and a CSV of labeled pairs:

```csv
file_x,file_y,decision
anna_2019.jpg,anna_2024.jpg,Yes
anna_2019.jpg,ben_2021.jpg,No
```

Use one main face per photo and include difficult pairs: different ages, side views, small faces, screenshots from videos. For **detection** it needs a folder of group photos, optionally with a `counts.json` of how many faces each one has (`{"party.jpg": 12}`).

```bash
uv run python scripts/face_benchmark.py \
  --model buffalo_l=./buffalo_l --model adaface=./my-model --model yolo=./my-yolo \
  --pairs pairs.csv --images ./people \
  --groups ./group-photos --reference-models buffalo_l,yolo \
  --output report.md
```

How to read the report:

- **d′** is how far apart the similarities of same-person and different-person pairs are; higher is better and it keeps telling models apart when all of them get everything right.
- **at Immich's default threshold** shows how many same-person pairs would be grouped together, and how many different-person pairs would be merged by mistake. A model may need a different _maximum recognition distance_ than the default.
- For detection, compare how many faces each model finds and how many of the faces the other detectors agree on. `--reference-models` should only list different networks: two sizes of the same detector always agree with each other.

## Step 4: Use the model in Immich

Make the model available to the machine learning container in one of two ways:

- **Hugging Face:** upload the folder with `hf upload <owner>/<repository> ./my-model` and use `<owner>/<repository>` as the model name. The container downloads it on first use.
- **Locally:** copy the folder into the model cache volume as `/cache/facial-recognition/<owner>--<repository>/`, e.g. `/cache/facial-recognition/local--adaface/detection/model.onnx`, and use `local/adaface` as the model name. A model that is already in the cache is not downloaded. With Rockchip (RKNN) or ARM NN acceleration, the container first looks for a model in that format and tries to download it, so use Hugging Face there.

Then, in Administration > Settings > Machine Learning Settings > Facial Recognition:

1. Set the **facial recognition model** and/or the **face detection model** to the name.
2. Adjust the **minimum detection score** if the detector scores differently (e.g. ~0.5 for YOLOv8n-face).
3. After changing the facial recognition model, re-run **Face Detection** for _all_ assets in Administration > Jobs.
