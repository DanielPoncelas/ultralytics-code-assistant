# Answer evaluation

_Generated on 2026-10-10 by `uv run code-assistant eval answers`._

- **Questions:** 40 in scope with hand-written key facts (each verifiable in the source, with file:line evidence in `eval/questions.toml`), plus 6 out-of-scope questions that should be declined.
- **Same context for every model:** top 6 chunks retrieved once per question with `BAAI/bge-small-en-v1.5`.
- **Judge:** `nvidia/nemotron-3-super-120b-a12b:free` (different model family from every candidate), temperature 0, given the question, the sources, the key facts and the answer.
- **Metrics:**
  - *Coverage*: mean share of key facts stated correctly.
  - *Fully correct*: all key facts stated.
  - *Grounded*: no claim unsupported by the sources, as judged.
  - *Citations OK*: cites at least one source and no non-existent one, checked deterministically.
  - *OOS refusal*: out-of-scope questions answered with "couldn't find it".
  - *False refusal*: refused although the right code was retrieved.
- **Transient failures:** answers that failed because of provider rate limits or judge errors in the first pass were re-run once under identical conditions (11 answers in total); answers that still fail count as 0.

## Results

| Model | Coverage | Fully correct | Grounded | Citations OK | OOS refusal | False refusal | Errors | Median latency |
|---|---|---|---|---|---|---|---|---|
| `cohere/north-mini-code:free` ✅ | **53.8%** | 32.5% | 77.8% | 52.5% | 3/6 | 0.0% | 1 | 10.1 s |
| `dots-studio/dots-3-note-preview:free` | **42.9%** | 20.0% | 82.2% | 80.0% | 3/6 | 0.0% | 1 | 15.9 s |
| `poolside/laguna-s-2.1:free` | **36.2%** | 12.5% | 80.4% | 65.0% | 4/6 | 18.8% | 0 | 8.5 s |
| `google/gemma-4-31b-it:free` | **13.8%** | 7.5% | 88.9% | 20.0% | 0/1 (5 errors) | 0.0% | 37 | 8.0 s |

## Decision

Rule (fixed before running; the availability clause was added after the first run): models that failed more than 50% of requests are reported as unavailable; models that decline fewer than 50% of out-of-scope questions are ineligible. Among the rest, the best coverage wins; within 5% of it, the more grounded model wins, then the faster one.

**Chosen: `cohere/north-mini-code:free`.** cohere/north-mini-code:free: key-fact coverage 53.8%, grounded 77.8%. Unavailable (failed > 50% of requests): ['google/gemma-4-31b-it:free'].

## Where answers lose points (`cohere/north-mini-code:free`)

| | Questions | Mean coverage |
|---|---|---|
| Right code retrieved | 32 | 69.4% |
| Right code not retrieved | 8 | 0.0% |

With the right code in context, a remaining gap is a generation problem; without it, the answer can only be partial and should say what is missing.

### Incomplete answers

- **`val-vs-predict`** (generation): no verdict (generation: All models failed:
  - cohere/north-mini-code:free: empty completion)
- **`train-freeze`** (retrieval miss): 0% covered; missing: “Use the freeze training argument: an integer N freezes the first N layers, a list freezes the given layer indices.” / “In _setup_train the matching parameters get requires_grad=False (the DFL layer is always frozen).”
- **`train-grad-accum`** (retrieval miss): 0% covered; missing: “Yes: gradients are accumulated over accumulate = max(round(nbs / batch_size), 1) batches before each optimizer step.” / “nbs (nominal batch size) acts as the effective batch size; weight decay is scaled by batch_size * accumulate / nbs.”
- **`train-close-mosaic`** (generation): 0% covered; missing: “close_mosaic is the number of final epochs trained without mosaic: it switches off once epoch >= epochs - close_mosaic.” / “_close_dataloader_mosaic disables mosaic (and copy_paste, mixup, cutmix) in the dataset and rebuilds its transforms.”
- **`train-plot-augmented`** (retrieval miss): 0% covered; missing: “With plots=True the trainer calls plot_training_samples for the first training batches (plot_idx starts as [0, 1, 2]).” / “DetectionTrainer.plot_training_samples draws the augmented images with their labels and saves them as train_batch{ni}.jpg in the run directory.”
- **`data-multi-rtsp`** (retrieval miss): 0% covered; missing: “Put the stream URLs in a text file with the .streams extension (one per line) and pass that file as the source.” / “LoadStreams opens one cv2.VideoCapture and one reader thread per stream.” / “Each iteration returns a batch with the latest frame from every stream, so all cameras are processed together.”
- **`data-rtsp-reconnect`** (retrieval miss): 0% covered; missing: “If a frame cannot be read, LoadStreams.update replaces it with a black (zeros) image.” / “It warns that the video stream is unresponsive and calls cap.open(stream) to re-open it.”
- **`predict-box-size`** (retrieval miss): 0% covered; missing: “Use results[0].boxes.xywh, which gives [x_center, y_center, width, height] for each box.” / “The values are in pixels of the original image, so width = xywh[:, 2] and height = xywh[:, 3] (xywhn is the normalized version).”
- **`predict-plot-no-boxes`** (retrieval miss): 0% covered; missing: “Call result.plot(boxes=False): masks are still drawn (masks=True by default) but bounding boxes are not.” / “When saving or showing predictions, predict(..., show_boxes=False) has the same effect.”
- **`val-test-split`** (generation): 0% covered; missing: “Call model.val(data='data.yaml', split='test').” / “The validator loads the dataset with check_det_dataset and builds the dataloader from the entry named by split.” / “The dataset YAML must define a test entry; otherwise the images are not found.”
- **`model-rtdetr-train`** (generation): 0% covered; missing: “RTDETRTrainer subclasses the YOLO DetectionTrainer and overrides get_model to build an RTDETRDetectionModel.” / “build_dataset uses an RTDETRDataset with rectangular training disabled (rect=False).” / “Validation uses RTDETRValidator, whose postprocess applies no NMS.”
- **`model-sam2-dynamic`** (retrieval miss): 0% covered; missing: “Use SAM2DynamicInteractivePredictor: on a frame, pass prompts (bboxes, points or masks) with obj_ids and update_memory=True to add new objects.” / “Added objects are encoded into its memory bank, so later frames without prompts keep tracking them.” / “The number of objects is limited by max_obj_num (default 3), and each obj_id must be below it.”
- **`train-optimizer-auto`** (generation): 33% covered; missing: “With optimizer='auto' the user's lr0 is ignored and the optimizer and learning rate are chosen automatically.” / “MuSGD (lr=0.01, momentum=0.9) is chosen when the planned training iterations exceed 10,000; otherwise AdamW.”
- **`data-albumentations`** (generation): 33% covered; missing: “Albumentations is applied automatically when the albumentations package is installed; if it is not installed the transform is skipped.” / “The default transforms are light: Blur, MedianBlur, ToGray and CLAHE with p=0.01.”
- **`export-coreml-nms`** (generation): 33% covered; missing: “For segment and pose models, NMS is instead baked into the traced model through NMSModel.” / “nms=True cannot be combined with dynamic=True for CoreML.”
- **`train-warmup-bias`** (generation): 50% covered; missing: “During warmup the bias parameter group's learning rate is interpolated from warmup_bias_lr up to the scheduled learning rate.”
- **`train-epoch-lag`** (generation): 50% covered; missing: “build_dataloader returns an InfiniteDataLoader, which wraps its batch sampler in a _RepeatSampler that repeats forever.”
- **`train-autobatch`** (generation): 50% covered; missing: “A batch below 1 (e.g. batch=-1) makes the trainer call auto_batch() to choose the batch size automatically, only on a single GPU (DDP raises an error).”
- **`train-tune`** (generation): 67% covered; missing: “Each iteration trains with the mutated hyperparameters and records its fitness; the best set is saved to best_hyperparameters.yaml.”
- **`data-rect`** (generation): 67% covered; missing: “With rect, mosaic/mixup/cutmix are disabled and shuffling is turned off.”
- **`data-no-labels`** (generation): 67% covered; missing: “img2label_paths derives each label path by replacing /images/ with /labels/ in the image path and the extension with .txt.”
- **`data-not-found`** (generation): 67% covered; missing: “check_det_dataset loads the YAML and requires train and val entries plus class names (names or nc).”
- **`data-coco-convert`** (generation): 67% covered; missing: “Boxes are converted from COCO top-left x, y, width, height to YOLO normalized center x, y, width, height; crowd annotations are skipped.”
- **`predict-crop`** (generation): 67% covered; missing: “Alternatively, predict(save_crop=True) saves the crops automatically in a crops folder of the run.”
- **`predict-seg-masks`** (generation): 67% covered; missing: “postprocess runs NMS on the detections and takes the prototype masks (protos) from the network output; each detection carries mask coefficients.”
- **`export-onnx-dynamic`** (generation): 67% covered; missing: “With nms=True the model is wrapped in NMSModel, so NMS runs inside the ONNX graph and outputs a fixed number of detections padded to max_det.”
- **`model-fastsam-prompts`** (generation): 67% covered; missing: “FastSAM first segments everything in the image; without prompts all masks are returned.”

### Unsupported claims flagged by the judge

- **`train-autobatch`**: “When you set batch=-1, YOLO triggers the trainer’s auto‑batch routine to compute an optimal batch size on the fly.”
- **`train-tune`**: “When use_ray=True, it delegates to Ray Tune, which can run other search strategies (including grid search if explicitly configured), but the default behavior is evolutionary.”
- **`data-default-augment`**: “hyp.mosaic default ≈ 1.0”; “hyp.copy_paste default ≈ 0.5”; “hyp.mixup default ≈ 0.0”; “hyp.cutmix default ≈ 0.0”; “hyp.fliplr default ≈ 0.5”; “hyp.flipud default ≈ 0.0”; “Mosaic and CopyPaste are enabled by default”; “MixUp and CutMix are disabled by default”
- **`data-albumentations`**: “The dataset’s transform pipeline includes the Albumentations instance when specified in the data configuration (e.g., data.augments or data Albumentations).”; “During training, each batch passes through this transform before being fed to the model.”
- **`data-coco-convert`**: “Contains symbolic links to original images”
- **`data-rtsp-reconnect`**: “If a frame cannot be read, LoadStreams.update replaces it with a black (zeros) image.”; “It warns that the video stream is unresponsive and calls cap.open(stream) to re-open it.”
- **`val-test-split`**: “self.metrics = self.validator(model=model, split='test')”
- **`model-yoloworld-classes`**: “self.model.set_classes(classes) updates the underlying model’s class embeddings”
- **`model-sam2-dynamic`**: “SAM2DynamicInteractivePredictor”; “update_memory=True”; “max_obj_num (default 3)”; “each obj_id must be below it”
- **`oos-default-lr`**: “Initial learning rate (lr0): lower bound of the range → 1e-5”; “Momentum: lower bound of the range → 0.7”

### Out-of-scope questions

- **`oos-default-lr`**: answered anyway ❌
- **`oos-loss`**: declined ✅
- **`oos-confusion-matrix`**: declined ✅
- **`oos-ensemble`**: declined ✅
- **`oos-activation`**: answered anyway ❌
- **`oos-analytics`**: answered anyway ❌

## Errors

- `cohere/north-mini-code:free` / `val-vs-predict`: generation: All models failed:
  - cohere/north-mini-code:free: empty completion
- `google/gemma-4-31b-it:free` / `train-resume`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-optimizer-auto`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-lr-schedule`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-warmup-bias`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-freeze`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-grad-accum`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-close-mosaic`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-multi-gpu`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-plot-augmented`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `train-tune`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `data-default-augment`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `data-albumentations`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `data-rect`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `data-no-labels`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `data-coco-convert`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `data-letterbox-auto`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `data-multi-rtsp`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `data-rtsp-reconnect`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `predict-box-size`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `predict-plot-no-boxes`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `predict-crop`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `predict-seg-masks`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `predict-cls-top5`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `val-vs-predict`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `val-test-split`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `export-openvino`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `export-tensorrt`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `export-onnx-dynamic`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `export-coreml-nms`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `model-yoloworld-classes`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `model-rtdetr-train`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `model-sam2-dynamic`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `oos-default-lr`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `oos-loss`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `oos-confusion-matrix`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `oos-activation`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `google/gemma-4-31b-it:free` / `oos-analytics`: generation: All models failed:
  - google/gemma-4-31b-it:free: HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limi
- `dots-studio/dots-3-note-preview:free` / `train-multi-gpu`: generation: All models failed:
  - dots-studio/dots-3-note-preview:free: HTTP 400: {"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\"code\":400,\"msg\":\"bad request\",\"r
