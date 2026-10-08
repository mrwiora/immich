# Face model benchmark — first results

Run on 2026-10-08 on branch `claude/configurable-face-models` with
`machine-learning/scripts/face_benchmark.py`, CPU only (4 cores, ONNX Runtime), every model run
through Immich's own `FaceDetector` / `FaceRecognizer` with a `model.json` where needed.

## Summary

| Question                                         | Answer from this run                                                                                                                                                                                                                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Does anything recognize better than `buffalo_l`? | **Yes: `antelopev2`** (already built into Immich) separates people best in every condition, including small, blurred and video-like faces. **AdaFace IR-101** is also better than `buffalo_l`, but below `antelopev2` at the same cost.                                                       |
| What is it worth?                                | At Immich's default threshold, `antelopev2` matches 91% of same-person pairs on video-like frames vs 75% for `buffalo_l`, and 94% vs 79% for 20 px faces. Recognition takes ~1.8× longer per face (110 vs 63 ms).                                                                             |
| Is there a cheaper model that is as good?        | **EdgeFace-Base** comes close to `buffalo_l` (slightly lower separation) at ~4× the speed (17 vs 63 ms per face). **EdgeFace-S** is a good replacement for `buffalo_s` on small/low-power servers.                                                                                            |
| Does any detector beat SCRFD-10G (`buffalo_l`)?  | Not overall. **YOLOv8n-face** finds more partly visible faces in close-ups (7 of 7 vs 4 in an elevator selfie) but far fewer small faces in crowds, and its scores run lower, so it needs a lower `minScore` than Immich's 0.7.                                                               |
| What would find more faces?                      | A **larger detector input**: SCRFD-10G at 1280 px instead of 640 px finds ~3× more (verified real) faces in a crowd, but misses very large close-up faces. Running 640 and 1280 and merging the results (multi-scale) is the most promising detection improvement, at ~4× the detection time. |
| Does the `model.json` color order matter?        | Yes. AdaFace fed RGB instead of the BGR it was trained on loses separation everywhere; for 14 px faces, same-person matches at Immich's threshold drop from 54% to 27%.                                                                                                                       |

### Caveats

- **Small, easy data.** 61 photos of 13 well-known people (the DeepFace test set), likely seen in
  training by some models. No model wrongly matched two different people at Immich's threshold,
  so false merges could not be compared. The standard benchmarks (LFW, CFP-FP, AgeDB, IJB-C,
  WIDER FACE) could not be downloaded from the development environment and remain to be run.
- **Synthetic degradations.** Small faces, blur, JPEG and low light are simulated on the second image of
  each pair; real video frames may behave differently.
- **Detection reference.** Group photos have no box annotations; the reference is every face that at
  least two of SCRFD-10G, SCRFD-500M and YOLOv8n-face agree on at 640 px, plus hand counts for the
  smaller photos. Faces only one detector finds count as "found" but not as "reference found".
- **Similarity scales differ per model.** The share of same-person pairs matched at Immich's
  `maxDistance` 0.5 depends on how a model spreads its similarities, so a model may need its own
  threshold (d′ and the match rate at 0.1% false matches do not depend on it).
- **Speed** is from one CPU, partly with two benchmarks running in parallel; compare ratios, not
  absolute numbers.

### Published numbers (from the model authors)

| Model                                   | Training data | IJB-C TAR@FAR=1e-4 | Other                                   |
| --------------------------------------- | ------------- | ------------------ | --------------------------------------- |
| `buffalo_l` recognition (ArcFace R50)   | WebFace600K   | 97.25              | LFW 99.83, CFP-FP 99.33, AgeDB-30 98.23 |
| `buffalo_s` recognition (MobileFaceNet) | WebFace600K   | 95.02              | LFW 99.70, CFP-FP 98.00, AgeDB-30 96.58 |
| AdaFace IR-50                           | WebFace4M     | 96.98              |                                         |
| AdaFace IR-101                          | WebFace12M    | 97.66              |                                         |
| EdgeFace-Base                           |               |                    | LFW 99.83, CFP-FP 97.01, AgeDB-30 97.60 |
| EdgeFace-S (γ=0.5)                      |               |                    | LFW 99.78, CFP-FP 95.74, AgeDB-30 97.03 |

| Detector                                | WIDER FACE val AP easy / medium / hard |
| --------------------------------------- | -------------------------------------- |
| SCRFD-10G with keypoints (`buffalo_l`)  | 95.40 / 94.01 / 82.80                  |
| SCRFD-500M with keypoints (`buffalo_s`) | 90.97 / 88.44 / 69.49                  |
| YOLOv8n-face                            | 94.6 / 92.3 / 79.6                     |

Sources: the InsightFace model zoo README, and the READMEs of yakhyo/adaface-onnx,
yakhyo/edgeface-onnx and yakhyo/yolov8-face-onnx-inference.

## Data

- Recognition: `tests/unit/dataset` of [serengil/deepface](https://github.com/serengil/deepface),
  labels from its `master.csv` and `face-recognition-pivot.csv`, extended transitively: 13 people,
  61 photos, 1692 pairs (140 same person). Faces found by `buffalo_l`'s detector for every
  recognizer.
- Detection: 8 photos from deepface, insightface, face_recognition, yolov8-face-onnx-inference and
  the Ultralytics assets, from 2 faces up to a crowd of several hundred.

## Recognition results

1692 pairs (140 same person) of 61 images, faces found by `buffalo_l`.

Each cell: d′ (separation of same/different person similarities, higher is better) · same-person pairs matched at a 0.1% false match rate · at Immich's default threshold (cosine distance ≤ 0.5): same-person pairs matched / different-person pairs wrongly matched.

| Degradation of the 2nd image          | buffalo_l                | buffalo_s                | antelopev2                 | adaface_ir50             | adaface_ir101              | adaface_ir101_rgb         | edgeface_base            | edgeface_s                |
| ------------------------------------- | ------------------------ | ------------------------ | -------------------------- | ------------------------ | -------------------------- | ------------------------- | ------------------------ | ------------------------- |
| original                              | 9.81 · 100% · 99% / 0.0% | 7.88 · 100% · 94% / 0.0% | 12.27 · 100% · 100% / 0.0% | 8.76 · 100% · 98% / 0.0% | 10.80 · 100% · 100% / 0.0% | 10.09 · 100% · 99% / 0.0% | 8.99 · 100% · 98% / 0.0% | 7.47 · 100% · 100% / 0.0% |
| face 32px                             | 9.18 · 100% · 99% / 0.0% | 7.43 · 100% · 89% / 0.0% | 11.31 · 100% · 100% / 0.0% | 8.29 · 100% · 94% / 0.0% | 10.46 · 100% · 99% / 0.0%  | 9.45 · 100% · 96% / 0.0%  | 8.30 · 100% · 98% / 0.0% | 7.35 · 100% · 100% / 0.0% |
| face 20px                             | 7.57 · 100% · 79% / 0.0% | 5.73 · 97% · 59% / 0.0%  | 8.94 · 100% · 94% / 0.0%   | 6.79 · 100% · 84% / 0.0% | 8.45 · 100% · 90% / 0.0%   | 7.28 · 100% · 77% / 0.0%  | 6.73 · 100% · 82% / 0.0% | 5.87 · 100% · 89% / 0.0%  |
| face 14px                             | 6.11 · 100% · 34% / 0.0% | 4.44 · 88% · 11% / 0.0%  | 6.09 · 99% · 27% / 0.0%    | 5.95 · 99% · 55% / 0.0%  | 6.56 · 100% · 54% / 0.0%   | 5.40 · 99% · 27% / 0.0%   | 5.31 · 100% · 46% / 0.0% | 4.77 · 93% · 63% / 0.0%   |
| blur                                  | 8.60 · 100% · 94% / 0.0% | 6.35 · 100% · 71% / 0.0% | 10.62 · 100% · 99% / 0.0%  | 7.46 · 100% · 88% / 0.0% | 9.90 · 100% · 98% / 0.0%   | 8.61 · 100% · 86% / 0.0%  | 7.86 · 100% · 94% / 0.0% | 6.94 · 100% · 99% / 0.0%  |
| motion blur                           | 7.83 · 100% · 83% / 0.0% | 6.47 · 99% · 75% / 0.0%  | 9.93 · 100% · 99% / 0.0%   | 7.34 · 100% · 84% / 0.0% | 8.94 · 100% · 94% / 0.0%   | 7.42 · 100% · 83% / 0.0%  | 7.35 · 100% · 89% / 0.0% | 6.49 · 100% · 94% / 0.0%  |
| jpeg q10                              | 8.83 · 100% · 97% / 0.0% | 7.26 · 100% · 89% / 0.0% | 11.96 · 100% · 100% / 0.0% | 8.02 · 100% · 94% / 0.0% | 9.95 · 100% · 99% / 0.0%   | 8.86 · 100% · 96% / 0.0%  | 8.50 · 100% · 98% / 0.0% | 7.17 · 100% · 99% / 0.0%  |
| low light                             | 9.71 · 100% · 99% / 0.0% | 7.82 · 100% · 91% / 0.0% | 11.96 · 100% · 100% / 0.0% | 8.53 · 100% · 95% / 0.0% | 10.68 · 100% · 99% / 0.0%  | 9.87 · 100% · 97% / 0.0%  | 8.67 · 100% · 96% / 0.0% | 7.44 · 100% · 100% / 0.0% |
| video frame (24px, motion blur, JPEG) | 7.26 · 100% · 75% / 0.0% | 5.38 · 98% · 52% / 0.0%  | 8.79 · 100% · 91% / 0.0%   | 6.28 · 100% · 76% / 0.0% | 7.66 · 100% · 81% / 0.0%   | 6.78 · 100% · 73% / 0.0%  | 6.55 · 100% · 83% / 0.0% | 6.12 · 100% · 91% / 0.0%  |

## Detection results at minimum score 0.7 (Immich's default)

`_1280` variants use `{"input_size": 1280}`; `@ 0.5` is the photo downscaled to half its size.

| Image                            | Reference | Hand count | buffalo_l @ 1 | buffalo_s @ 1 | yolov8n_face @ 1 | buffalo_l_1280 @ 1 | buffalo_s_1280 @ 1 | yolov8n_face_1280 @ 1 | buffalo_l @ 0.5 | buffalo_s @ 0.5 | yolov8n_face @ 0.5 | buffalo_l_1280 @ 0.5 | buffalo_s_1280 @ 0.5 | yolov8n_face_1280 @ 0.5 |
| -------------------------------- | --------- | ---------- | ------------- | ------------- | ---------------- | ------------------ | ------------------ | --------------------- | --------------- | --------------- | ------------------ | -------------------- | -------------------- | ----------------------- |
| couple.jpg                       | 2         | 2          | 2 / 2         | 2 / 2         | 2 / 2            | 0 / 0              | 0 / 0              | 2 / 2                 | 2 / 2           | 2 / 2           | 2 / 2              | 0 / 0                | 0 / 0                | 2 / 2                   |
| face_recognition_two_people.jpg  | 2         | 2          | 2 / 2         | 2 / 2         | 2 / 2            | 2 / 2              | 2 / 2              | 2 / 2                 | 2 / 2           | 2 / 2           | 2 / 2              | 2 / 2                | 2 / 2                | 2 / 2                   |
| insightface_t1.jpg               | 6         | 6          | 6 / 6         | 6 / 6         | 6 / 6            | 6 / 6              | 6 / 6              | 6 / 6                 | 6 / 6           | 6 / 6           | 6 / 6              | 6 / 6                | 6 / 6                | 6 / 6                   |
| selfie-many-people.jpg           | 4         | 7          | 4 / 4         | 4 / 4         | 7 / 4            | 4 / 4              | 4 / 4              | 5 / 4                 | 4 / 4           | 4 / 4           | 7 / 4              | 4 / 4                | 5 / 4                | 6 / 4                   |
| ultralytics_bus.jpg              | 2         | 2          | 2 / 2         | 2 / 2         | 2 / 2            | 2 / 2              | 2 / 2              | 2 / 2                 | 2 / 2           | 2 / 2           | 2 / 2              | 2 / 2                | 2 / 2                | 2 / 2                   |
| ultralytics_zidane.jpg           | 2         | 2          | 1 / 1         | 2 / 2         | 2 / 2            | 1 / 1              | 1 / 1              | 2 / 2                 | 1 / 1           | 2 / 2           | 2 / 2              | 1 / 1                | 1 / 1                | 2 / 2                   |
| yolov8_large_selfi.jpg           | 67        |            | 107 / 67      | 64 / 63       | 47 / 44          | 333 / 67           | 231 / 66           | 168 / 67              | 109 / 66        | 62 / 60         | 48 / 44            | 299 / 67             | 223 / 66             | 146 / 67                |
| yolov8_test.jpg                  | 19        |            | 22 / 19       | 17 / 17       | 20 / 19          | 19 / 18            | 19 / 18            | 22 / 19               | 21 / 19         | 16 / 16         | 19 / 19            | 19 / 18              | 18 / 18              | 22 / 19                 |
| **share of the reference found** | 104       |            | 99.0%         | 94.2%         | 77.9%            | 96.2%              | 95.2%              | 100.0%                | 98.1%           | 90.4%           | 77.9%              | 96.2%                | 95.2%                | 100.0%                  |

## Detection results at minimum score 0.5

| Image                            | Reference | Hand count | buffalo_l @ 1 | buffalo_s @ 1 | yolov8n_face @ 1 | buffalo_l_1280 @ 1 | buffalo_s_1280 @ 1 | yolov8n_face_1280 @ 1 | buffalo_l @ 0.5 | buffalo_s @ 0.5 | yolov8n_face @ 0.5 | buffalo_l_1280 @ 0.5 | buffalo_s_1280 @ 0.5 | yolov8n_face_1280 @ 0.5 |
| -------------------------------- | --------- | ---------- | ------------- | ------------- | ---------------- | ------------------ | ------------------ | --------------------- | --------------- | --------------- | ------------------ | -------------------- | -------------------- | ----------------------- |
| couple.jpg                       | 2         | 2          | 2 / 2         | 2 / 2         | 2 / 2            | 2 / 2              | 1 / 1              | 2 / 2                 | 2 / 2           | 2 / 2           | 2 / 2              | 2 / 2                | 1 / 1                | 2 / 2                   |
| face_recognition_two_people.jpg  | 2         | 2          | 2 / 2         | 2 / 2         | 2 / 2            | 2 / 2              | 2 / 2              | 2 / 2                 | 2 / 2           | 2 / 2           | 2 / 2              | 2 / 2                | 2 / 2                | 2 / 2                   |
| insightface_t1.jpg               | 6         | 6          | 6 / 6         | 6 / 6         | 6 / 6            | 6 / 6              | 6 / 6              | 6 / 6                 | 6 / 6           | 6 / 6           | 6 / 6              | 6 / 6                | 6 / 6                | 6 / 6                   |
| selfie-many-people.jpg           | 6         | 7          | 6 / 6         | 6 / 6         | 7 / 6            | 5 / 5              | 6 / 5              | 7 / 6                 | 6 / 6           | 6 / 6           | 7 / 6              | 5 / 5                | 6 / 5                | 7 / 6                   |
| ultralytics_bus.jpg              | 2         | 2          | 2 / 2         | 2 / 2         | 2 / 2            | 3 / 2              | 2 / 2              | 2 / 2                 | 2 / 2           | 2 / 2           | 2 / 2              | 2 / 2                | 2 / 2                | 2 / 2                   |
| ultralytics_zidane.jpg           | 2         | 2          | 2 / 2         | 2 / 2         | 2 / 2            | 2 / 2              | 2 / 2              | 2 / 2                 | 2 / 2           | 2 / 2           | 2 / 2              | 2 / 2                | 2 / 2                | 2 / 2                   |
| yolov8_large_selfi.jpg           | 139       |            | 202 / 139     | 116 / 110     | 157 / 129        | 490 / 139          | 381 / 138          | 435 / 139             | 199 / 138       | 120 / 111       | 160 / 128          | 463 / 139            | 368 / 138            | 409 / 139               |
| yolov8_test.jpg                  | 25        |            | 26 / 25       | 23 / 23       | 24 / 24          | 25 / 25            | 24 / 24            | 27 / 24               | 27 / 25         | 23 / 23         | 26 / 25            | 26 / 25              | 24 / 24              | 27 / 24                 |
| **share of the reference found** | 184       |            | 100.0%        | 83.2%         | 94.0%            | 99.5%              | 97.8%              | 99.5%                 | 99.5%           | 83.7%           | 94.0%              | 99.5%                | 97.8%                | 99.5%                   |

## Speed

Median on this CPU; compare ratios.

| Recognition model         | Per face |
| ------------------------- | -------- |
| buffalo_l (ArcFace R50)   | 63 ms    |
| buffalo_s (MobileFaceNet) | 14 ms    |
| antelopev2 (ArcFace R100) | 110 ms   |
| AdaFace IR-50             | 68 ms    |
| AdaFace IR-101            | 115 ms   |
| EdgeFace-Base             | 17 ms    |
| EdgeFace-S                | 9 ms     |

| Detector                          | Per image at 640 px | Per image at 1280 px |
| --------------------------------- | ------------------- | -------------------- |
| SCRFD-10G (buffalo_l, antelopev2) | 153 ms              | 638 ms               |
| SCRFD-500M (buffalo_s)            | 30 ms               | 84 ms                |
| YOLOv8n-face                      | 82 ms               | 224 ms               |
