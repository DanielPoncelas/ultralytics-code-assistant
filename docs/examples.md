# Example answers

Generated on 2026-10-10 with `uv run code-assistant examples` (index: Ultralytics `v8.4.175`, embeddings: `BAAI/bge-small-en-v1.5`, top 6 sources). Answers are reproduced exactly as generated; LLM output varies between runs.

## 1. How do the lr0 and lrf hyperparameters work? Is the learning rate decayed linearly or with a cosine schedule?

*Source: [issue #2849](https://github.com/ultralytics/ultralytics/issues/2849) · answered by `cohere/north-mini-code:free` in 5.2 s*

**Answer**

- **`lr0`** – the *initial* learning rate for the optimizer.  
- **`lrf`** – the *final* learning‑rate **fraction** (the final LR = `lr0 * lrf`).  

The decay **shape** (linear vs. cosine) is controlled by the trainer’s `cos_lr` flag:

| `cos_lr` value | Schedule used | Formula (from source [1]) |
|----------------|---------------|---------------------------|
| `True`         | **Cosine**    | `self.lf = one_cycle(1, self.args.lrf, self.epochs)` |
| `False`        | **Linear**    | `self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf` |

So the learning‑rate starts at `lr0`, ends at `lr0 × lrf`, and follows either a cosine curve (if `cos_lr=True`) or a straight‑line decay (if `cos_lr=False`). The schedule is applied each epoch via `self.scheduler.step()` (see source [3]).

**Key points**

- `lr0` and `lrf` are part of the default hyperparameter space defined in the Tuner (source [2]):  
  ```python
  "lr0": (1e-5, 1e-2),      # initial LR
  "lrf": (0.01, 1.0),      # final LR fraction
  ```
- The actual LR values are scaled by `self.lf(epoch)` during training (source [3]), which implements the chosen schedule.

**Retrieved sources** (bold = cited in the answer):

1. **[`BaseTrainer._setup_scheduler`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L268-L274)** — `ultralytics/engine/trainer.py` L268-274
2. **[`Tuner.__init__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/tuner.py#L79-L142)** — `ultralytics/engine/tuner.py` L79-142
3. **[`BaseTrainer._do_train`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L466-L542)** — `ultralytics/engine/trainer.py` L466-542
4. [`Model.train`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L807-L877) — `ultralytics/engine/model.py` L807-877
5. [`TinyViT.set_layer_lr_decay`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/sam/modules/tiny_encoder.py#L900-L933) — `ultralytics/models/sam/modules/tiny_encoder.py` L900-933
6. [`TinyViT.__init__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/sam/modules/tiny_encoder.py#L787-L864) — `ultralytics/models/sam/modules/tiny_encoder.py` L787-864

## 2. What does close_mosaic do, and at which point of training is mosaic augmentation switched off?

*Source: [issue #14886](https://github.com/ultralytics/ultralytics/issues/14886) · answered by `cohere/north-mini-code:free` in 8.4 s*

**What `close_mosaic` does**

- **Disables mosaic (and related) augmentations** – it sets `hyp.mosaic`, `hyp.copy_paste`, `hyp.mixup` and `hyp.cutmix` to `0.0` and rebuilds the dataset transforms so those augmentations are no longer applied.  
  *[YOLODataset.close_mosaic]*[2]  

- **Updates the dataloader** – it also forces the dataset’s `mosaic` flag to `False` (if present) and calls the dataset’s own `close_mosaic` method (if it exists).  
  *[BaseTrainer._close_dataloader_mosaic]*[1]  

- **For concatenated datasets** – `YOLOConcatDataset.close_mosaic` propagates the call to each child dataset.  
  *[YOLOConcatDataset.close_mosaic]*[3]  

- **YOLO‑E specific extension** – `YOLOEVPTrainer._close_dataloader_mosaic` first calls the parent `_close_dataloader_mosaic` and then appends a `LoadVisualPrompt` transform to the dataset’s transforms.  
  *[YOLOEVPTrainer._close_dataloader_mosaic]*[5]

**When mosaic is switched off**

Mosaic (and the related augmentations) are turned off **at the start of the final `close_mosaic` epochs**:

```python
if not mosaic_closed and epoch >= self.epochs - self.args.close_mosaic:
    mosaic_closed = True
    self._close_dataloader_mosaic()
    self.train_loader.reset()
```

Thus, if `args.close_mosaic = N`, mosaic is used for the first `epochs - N` epochs and is disabled beginning with epoch `epochs - N`.  
*[BaseTrainer._do_train]*[6]

**Retrieved sources** (bold = cited in the answer):

1. **[`BaseTrainer._close_dataloader_mosaic`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L1143-L1149)** — `ultralytics/engine/trainer.py` L1143-1149
2. **[`YOLODataset.close_mosaic`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/dataset.py#L386-L396)** — `ultralytics/data/dataset.py` L386-396
3. **[`YOLOConcatDataset.close_mosaic`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/dataset.py#L877-L886)** — `ultralytics/data/dataset.py` L877-886
4. [`BaseTrainer`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L79-L1242) — `ultralytics/engine/trainer.py` L79-1242
5. **[`YOLOEVPTrainer._close_dataloader_mosaic`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/yoloe/train.py#L274-L281)** — `ultralytics/models/yolo/yoloe/train.py` L274-281
6. **[`BaseTrainer._do_train`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L466-L542)** — `ultralytics/engine/trainer.py` L466-542

## 3. Which data augmentations are applied by default when training a detection model?

*Source: [issue #2412](https://github.com/ultralytics/ultralytics/issues/2412) · answered by `cohere/north-mini-code:free` in 17.6 s*

**Default detection augmentations (YOLOv8)**  

When you call `model.train()` on a detection model, the training pipeline uses the transforms defined in `v8_transforms` (see source 1). By default (i.e., with the standard `hyp` values), the following augmentations are applied in order:

| Augmentation | Description | Source |
|--------------|-------------|--------|
| **Mosaic** | Combines 4 images into one, creating a mixed‑context training sample. | [1] |
| **RandomPerspective** (often called *affine*) | Applies random degrees, translation, scale, shear, and perspective distortion. | [1] |
| **CopyPaste** | Copies objects from one image to another (with a default `p=0.5`). | [1] |
| **MixUp** | Linear interpolation between an image and its label. | [1] |
| **CutMix** | Cuts a region from one image and pastes it into another. | [1] |
| **RandomHSV** | Randomly adjusts hue, saturation, and value. | [1] |
| **RandomFlip** (horizontal & vertical) | Random horizontal (`p=hyp.fliplr`) and vertical (`p=hyp.flipud`) flips. | [1] |
| **Albumentations** | Included in the pipeline but with `transforms=None` by default, so no extra transforms are applied unless you set `hyp.augmentations`. | [1] |

**Key points**

- The **Albumentations** wrapper is always present (so you can later inject custom transforms via `hyp.augmentations`), but it adds no extra operations unless you provide a list of transforms.
- All of the above are part of the `Compose` returned by `v8_transforms`, which is the default training pipeline for detection models (see `DetectionTrainer._build_train_pipeline` in source 5, which ultimately calls `v8_transforms`).

So, out‑of the box, a detection model sees **Mosaic → RandomPerspective → CopyPaste → MixUp → CutMix → RandomHSV → RandomFlip (H & V)** during each training iteration.

**Retrieved sources** (bold = cited in the answer):

1. **[`v8_transforms`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/augment.py#L2831-L2912)** — `ultralytics/data/augment.py` L2831-2912
2. [`Model.train`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L807-L877) — `ultralytics/engine/model.py` L807-877
3. [`classify_augmentations`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/augment.py#L2965-L3036) — `ultralytics/data/augment.py` L2965-3036
4. [`DetectionTrainer.build_dataset`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/detect/train.py#L64-L76) — `ultralytics/models/yolo/detect/train.py` L64-76
5. [`DetectionTrainer`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/detect/train.py#L24-L256) — `ultralytics/models/yolo/detect/train.py` L24-256
6. [`Format.__init__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/augment.py#L2356-L2393) — `ultralytics/data/augment.py` L2356-2393

## 4. How does the TensorRT export work? Does it go through ONNX first?

*Source: [issue #1225](https://github.com/ultralytics/ultralytics/issues/1225) · answered by `cohere/north-mini-code:free` in 2.7 s*

**Yes – TensorRT export goes through ONNX first.**

The `Exporter.export_engine` method (source 1) follows this flow:

```python
def export_engine(self, prefix=colorstr("TensorRT:")):
    # 1️⃣ Export to ONNX (required before TRT import)
    f_onnx = self.export_onnx()                     # ← creates .onnx file
    from ultralytics.utils.export.engine import onnx2engine

    assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"

    # 2️⃣ Convert ONNX → TensorRT engine
    f = self.file.with_suffix(".engine")           # output .engine file
    onnx2engine(
        f_onnx,
        f,
        self.args.workspace,
        self.args.quantize,
        self.args.dynamic,
        self.im.shape,
        dla=self.dla,
        dataset=self.get_int8_calibration_dataloader(prefix) if self.args.quantize == 8 and not self.qat else None,
        metadata=self.metadata,
        verbose=self.args.verbose,
        prefix=prefix,
    )
    return f
```

- **Step 1** – `self.export_onnx()` generates the intermediate ONNX model.  
- **Step 2** – `onnx2engine` (from `ultralytics.utils.export.engine`) converts that ONNX file into the final TensorRT engine (`.engine`).

The method also handles optional features like INT8 quantization, dynamic shapes, workspace size, DLA cores, and calibration data, but the core export pipeline always passes through ONNX.

**Retrieved sources** (bold = cited in the answer):

1. [`Exporter.export_engine`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L1437-L1460) — `ultralytics/engine/exporter.py` L1437-1460
2. [`Exporter.export_saved_model`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L1462-L1496) — `ultralytics/engine/exporter.py` L1462-1496
3. [`Exporter`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L530-L1868) — `ultralytics/engine/exporter.py` L530-1868
4. [`Exporter.__call__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L909-L975) — `ultralytics/engine/exporter.py` L909-975
5. [`Model.export`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L753-L805) — `ultralytics/engine/model.py` L753-805
6. [`export_formats`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L155-L285) — `ultralytics/engine/exporter.py` L155-285

## 5. How do I set my own class names for open-vocabulary detection with YOLO-World?

*Source: [issue #10038](https://github.com/ultralytics/ultralytics/issues/10038) · answered by `cohere/north-mini-code:free` in 4.1 s*

To set your own class names for open-vocabulary detection with YOLO-World, use the `set_classes` method:

```python
from ultralytics import YOLOWorld

# Load YOLO-World model
model = YOLOWorld("yolov8s-world.pt")

# Set custom class names for detection
model.set_classes(["person", "car", "bicycle"])
```

**Key details:**
- The `set_classes` method accepts a list of strings representing your desired categories
- It automatically drops any background class (represented by " ")
- After setting classes, the predictor is reset (`self.predictor = None`)
- This enables open-vocabulary detection for your specified classes

**Example usage:**
```python
# Set classes before inference
model.set_classes(["person", "dog", "cat", "bus"])

# Now the model will detect only these classes
results = model("image.jpg")
```

The method internally calls `self.model.set_classes(classes)` and updates the model's names attribute, making your custom classes available for detection.

**Retrieved sources** (bold = cited in the answer):

1. [`YOLOWorld`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/model.py#L148-L211) — `ultralytics/models/yolo/model.py` L148-211
2. [`YOLOWorld.set_classes`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/model.py#L202-L211) — `ultralytics/models/yolo/model.py` L202-211
3. [`WorldValidator`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/world/val.py#L12-L40) — `ultralytics/models/yolo/world/val.py` L12-40
4. [`YOLOWorld.__init__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/model.py#L173-L188) — `ultralytics/models/yolo/model.py` L173-188
5. [`WorldValidator.__call__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/world/val.py#L20-L40) — `ultralytics/models/yolo/world/val.py` L20-40
6. [`YOLOE.set_classes`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/model.py#L354-L369) — `ultralytics/models/yolo/model.py` L354-369
