# Example answers

Generated on 2026-10-10 with `uv run code-assistant examples` (index: Ultralytics `v8.4.175`, embeddings: `BAAI/bge-small-en-v1.5`, top 6 sources). Answers are reproduced exactly as generated; LLM output varies between runs.

## 1. My training run was interrupted at epoch 63. How do I resume it from the last checkpoint, and what state gets restored?

*Source: [issue #17492](https://github.com/ultralytics/ultralytics/issues/17492) · answered by `cohere/north-mini-code:free` in 24.0 s*

**How to resume from the interrupted run**

```bash
# From the project root
yolo train resume=model=path/to/last.pt   # or: yolo train resume=True
```

**What happens under the hood**

1. **`Model.train`** sees `args.get("resume")` is `True` and, if a valid checkpoint (`self.ckpt`) exists, sets `args["resume"]` and `args["data"]` to the checkpoint path.  
   → *[3]*

2. **`BaseTrainer.check_resume`** is called (via the trainer constructor). It:
   - Locates the latest checkpoint (`last.pt`) if `resume` is a path.  
   - Loads the checkpoint’s `args` (training configuration) and merges any allowed overrides.  
   - Sets `self.resume = True` and updates `self.args` with the checkpoint’s settings.  
   → *[2]*

3. **`BaseTrainer.resume_training`** is invoked (usually inside `trainer.train()`). It:
   - Reads `ckpt["epoch"]` → `start_epoch = epoch + 1` (so epoch 63 → start at epoch 64).  
   - Calls `self._load_checkpoint_state(ckpt)` which restores all training state.  
   - For distillation models, re‑initialises `model.criterion` and sets its `updates`.  
   - Stores the new `self.start_epoch`.  
   → *[1]*

**What state is restored from the checkpoint**

The checkpoint saved by `BaseTrainer.save_model` (`last.pt`, `best.pt`, `epochX.pt`) contains:

| Key                | Restored by `resume_training` (`_load_checkpoint_state`) |
|--------------------|----------------------------------------------------------|
| `epoch`            | `self.start_epoch` (next epoch to train)                |
| `best_fitness`     | `self.best_fitness`                                      |
| `stopper`          | `self.stopper` (best fitness / epoch)                   |
| `optimizer`        | Full optimizer state (learning rates, momentum, …)     |
| `scaler`           | Gradient‑scaler state (for mixed‑precision)             |
| `ema`              | Exponential moving‑average model weights                |
| `modelopt`         | Quantization state (QAT)                                 |
| `train_args`       | `self.args` (training hyper‑parameters)                  |
| `train_metrics`    | `self.metrics` (previous epoch metrics)                 |
| `train_results`    | `self.read_results_csv()` (CSV log)                      |
| `date`, `version`, `git`, `license`, `docs` | Metadata (not used for training) |

*For distillation models the `model.criterion` is rebuilt and its `updates` are set to `start_epoch‑1` so loss tracking continues correctly.*  
→ *[1][5]*

**Result**

- Training continues from **epoch 64** (the next epoch after the interruption).  
- Optimizer, scheduler, scaler, EMA weights, and all logged metrics are restored.  
- The model’s `criterion` (including any distillation‑specific logic) is re‑initialised with the correct update count.  

That’s the complete picture of resuming a stopped YOLO training run.

**Retrieved sources** (bold = cited in the answer):

1. **[`BaseTrainer.resume_training`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L1124-L1141)** — `ultralytics/engine/trainer.py` L1124-1141
2. **[`BaseTrainer.check_resume`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L1020-L1065)** — `ultralytics/engine/trainer.py` L1020-1065
3. **[`Model.train`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L875-L913)** — `ultralytics/engine/model.py` L875-913
4. [`BaseTrainer._handle_nan_recovery`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L1084-L1122) — `ultralytics/engine/trainer.py` L1084-1122
5. **[`BaseTrainer.save_model`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L764-L833)** — `ultralytics/engine/trainer.py` L764-833
6. [`_rewind`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/depth/calibrate.py#L38-L49) — `ultralytics/models/yolo/depth/calibrate.py` L38-49

## 2. Which data augmentations are applied by default when training a detection model?

*Source: [issue #2412](https://github.com/ultralytics/ultralytics/issues/2412) · answered by `cohere/north-mini-code:free` in 5.0 s*

**Default data augmentations for detection training**

When you train a detection model (e.g., YOLO object detection), the default augmentation pipeline is defined in `v8_transforms` (source 3). It applies the following transforms in order:

| Augmentation | Purpose | Applied by default? |
|--------------|---------|---------------------|
| **Mosaic** | Combines 4 images into one, increasing dataset diversity | ✅ |
| **RandomPerspective** (affine) | Random cropping, scaling, rotation, shear, perspective | ✅ |
| **CopyPaste** | Copies objects from one image to another for rare‑class augmentation | ✅ (if `hyp.copy_paste` > 0) |
| **MixUp** | Linear interpolation of images and targets | ✅ (if `hyp.mixup` > 0) |
| **CutMix** | Patches from other images are mixed in | ✅ (if `hyp.cutmix` > 0) |
| **Albumentations** | User‑specified custom transforms (via `hyp.augmentations`) | ✅ (if provided) |
| **RandomHSV** | Adjust hue, saturation, value for color variation | ✅ |
| **RandomFlip** (horizontal & vertical) | Random left‑right / up‑down flips (controlled by `hyp.fliplr` / `hyp.flipud`) | ✅ |

These transforms are composed into a single `Compose` object that is used by the detection trainer’s data loading pipeline (see `DetectionTrainer._build_train_pipeline` in source 2). Any custom augmentations you add via the `hyp.augmentations` list are inserted into the Albumentations step.

**Key points**

- The pipeline is **hard‑coded** in `v8_transforms` (source 3) and is the default for all YOLO detection tasks.
- You can adjust the strength of each augmentation via the `hyp` namespace (e.g., `hyp.mosaic`, `hyp.copy_paste`, `hyp.mixup`, `hyp.cutmix`, `hyp.fliplr`, `hyp.flipud`, `hyp.hsv_h`, etc.).
- If you need different augmentations (e.g., for classification), you would use `classify_augmentations` (source 5) instead.

**Example snippet (source 3)**

```python
transforms = v8_transforms(
    dataset,
    imgsz=640,
    hyp=hyp   # contains all the hyper‑parameters above
)
```

Thus, the **default detection augmentations** are **Mosaic → RandomPerspective → CopyPaste → MixUp → CutMix → Albumentations → RandomHSV → RandomFlip (horizontal & vertical)**.

**Retrieved sources** (bold = cited in the answer):

1. [`Model.train`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L807-L877) — `ultralytics/engine/model.py` L807-877
2. [`DetectionTrainer`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/detect/train.py#L24-L256) — `ultralytics/models/yolo/detect/train.py` L24-256
3. [`v8_transforms`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/augment.py#L2831-L2912) — `ultralytics/data/augment.py` L2831-2912
4. [`YOLOMultiModalDataset`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/dataset.py#L564-L629) — `ultralytics/data/dataset.py` L564-629
5. [`classify_augmentations`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/augment.py#L2965-L3036) — `ultralytics/data/augment.py` L2965-3036
6. [`MixUp`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/augment.py#L784-L883) — `ultralytics/data/augment.py` L784-883

## 3. Why do model.val() and model.predict() give different detections on the same images?

*Source: [issue #2783](https://github.com/ultralytics/ultralytics/issues/2783) · answered by `cohere/north-mini-code:free` in 13.9 s*

**Short answer:**  
`model.val()` and `model.predict()` use different default hyper‑parameters and post‑processing settings, so the same image can produce different detections.

**Why the difference?**  

| Aspect | `model.predict()` | `model.val()` (validator) |
|--------|-------------------|---------------------------|
| **Default confidence** | `conf = 0.25` (hard‑coded in `Model.predict`)【4】 | Validator uses its own `conf` (usually lower, e.g. `0.001`‑`0.05`) – not shown in the snippet but implied by the validator’s `self.args`. |
| **IOU / NMS** | May keep user‑provided `iou`/`agnostic_nms` (YOLOE sets `agnostic_nms=True` by default)【3】 | Validator typically uses class‑specific NMS and a different `iou` threshold. |
| **Maximum detections** | `max_det` is taken from `self.args.max_det` when plotting【1】 (validator‑specific). | `model.predict()` does not enforce a `max_det` in the predictor; the validator does. |
| **Rectangular inference** | `rect = True` by default【4】 | Validator often runs with `rect = False` (or a different `rect` setting). |
| **Source of thresholds** | All thresholds come from `args = {**self.overrides, **custom, **kwargs}`【4】 | Validator builds its own `args` (e.g., `self.args`) which can differ from the predictor’s defaults. |

Because `model.predict()` hard‑codes `conf=0.25`, `rect=True`, and (for YOLOE) `agnostic_nms=True`【3,4】, while the validation routine uses its own defaults (different `conf`, `iou`, `max_det`, NMS mode, etc.)【1,3,4】, the two pipelines apply different filtering and post‑processing to the raw model outputs. This leads to **different sets of bounding boxes** even when the same image and model are used.

**What you can do to make them match:**  
- Explicitly pass the same `conf`, `iou`, `max_det`, `agnostic_nms`, and `rect` values to both `model.val()` (via `self.overrides` or `args`) and `model.predict()`.  
- Ensure the validator’s `self.args` are aligned with the predictor’s `args` (e.g., set `conf=0.25` in the validator’s configuration).  

When the hyper‑parameters are identical, the detections from `model.val()` and `model.predict()` will be consistent.

**Retrieved sources** (bold = cited in the answer):

1. [`DetectionValidator.plot_predictions`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/detect/val.py#L451-L477) — `ultralytics/models/yolo/detect/val.py` L451-477
2. [`ClassificationValidator`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/classify/val.py#L18-L219) — `ultralytics/models/yolo/classify/val.py` L18-219
3. [`YOLOE.predict`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/model.py#L532-L567) — `ultralytics/models/yolo/model.py` L532-567
4. [`Model.predict`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L491-L555) — `ultralytics/engine/model.py` L491-555
5. [`DetectionValidator.plot_val_samples`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/detect/val.py#L436-L449) — `ultralytics/models/yolo/detect/val.py` L436-449
6. [`SemanticSegmentationValidator.plot_predictions`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/semantic/val.py#L181-L190) — `ultralytics/models/yolo/semantic/val.py` L181-190

## 4. How can I export a model to ONNX with a dynamic batch size and NMS included in the graph?

*Source: [issue #23339](https://github.com/ultralytics/ultralytics/issues/23339) · answered by `cohere/north-mini-code:free` in 4.0 s*

To export a YOLO model to ONNX with dynamic batch size and NMS included in the graph, use these parameters:

```python
model.export(
    format="onnx",    # ONNX format
    dynamic=True,     # Enable dynamic batch size
    nms=True,         # Include NMS in the graph
    batch=16,         # Maximum batch size (required when dynamic=True)
    simplify=True     # Optional: simplify the ONNX model
)
```

**Key points:**
- `dynamic=True` enables shape flexibility for batch dimension
- `nms=True` embeds NMS post-processing directly in the ONNX graph
- `batch=16` (or any value > 1) is required when using `dynamic=True` to specify the maximum batch size
- The NMS will be included as part of the exported ONNX model, reducing post-processing overhead during inference

The exported model will have dynamic input shapes for the batch dimension and contain the NMS operation within the ONNX graph itself.

**Retrieved sources** (bold = cited in the answer):

1. [`Exporter.export_onnx`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L1175-L1245) — `ultralytics/engine/exporter.py` L1175-1245
2. [`Exporter.__call__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L842-L911) — `ultralytics/engine/exporter.py` L842-911
3. [`Model.export`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L753-L805) — `ultralytics/engine/model.py` L753-805
4. [`Exporter.export_qnn`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L1662-L1682) — `ultralytics/engine/exporter.py` L1662-1682
5. [`NMSModel.forward`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L1957-L2023) — `ultralytics/engine/exporter.py` L1957-2023
6. [`Exporter.export_mnn`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L1333-L1344) — `ultralytics/engine/exporter.py` L1333-1344

## 5. What are the default values of the initial learning rate and momentum?

*Source: [issue #2849](https://github.com/ultralytics/ultralytics/issues/2849) · answered by `cohere/north-mini-code:free` in 5.9 s*

The default values are defined in the hyperparameter‑tuning space:

- **Initial learning rate (`lr0`)**: `1e‑5`  
- **Momentum**: `0.7` (the minimum value in the momentum range)

*Source: [1]*

**Retrieved sources** (bold = cited in the answer):

1. **[`Tuner.__init__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/tuner.py#L79-L142)** — `ultralytics/engine/tuner.py` L79-142
2. [`BaseTrainer.build_optimizer`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L1215-L1242) — `ultralytics/engine/trainer.py` L1215-1242
3. [`TinyViT.set_layer_lr_decay`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/sam/modules/tiny_encoder.py#L900-L933) — `ultralytics/models/sam/modules/tiny_encoder.py` L900-933
4. [`HungarianMatcher.__init__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/utils/ops.py#L44-L66) — `ultralytics/models/utils/ops.py` L44-66
5. [`Hiera.__init__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/sam/modules/encoders.py#L652-L736) — `ultralytics/models/sam/modules/encoders.py` L652-736
6. [`Model._val_data`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L1167-L1181) — `ultralytics/engine/model.py` L1167-1181
