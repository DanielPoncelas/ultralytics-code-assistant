# Judge review sheet

Blind review of 10 answers by `cohere/north-mini-code:free`, sampled at random (seed 0).

For each answer, tick `[x]` every key fact the answer states correctly (in any wording). Leave `[ ]` if it is missing or wrong. The judge's verdicts are deliberately not shown. When done, run `uv run code-assistant eval agreement`.

## 1. `predict-returns-list`

**Question:** I get AttributeError: 'list' object has no attribute 'boxes' after calling the model. What does prediction return?

> The model’s `predict()` method returns a **list of `Results` objects**. Each `Results` object has a `.boxes` attribute (a `Boxes` instance) that contains the detection data (coordinates, class IDs, scores, etc.).
> 
> **Why the error occurs**
> You tried to access `.boxes` on the list itself, e.g.:
> 
> ```python
> boxes = model.predict(...).boxes   # AttributeError: 'list' object has no attribute 'boxes'
> ```
> 
> The list does not have a `.boxes` attribute; only each individual `Results` object does.
> 
> **How to get the boxes**
> 
> ```python
> results = model.predict(...)          # list[Results]
> 
> # Iterate over all images
> for r in results:
>     boxes = r.boxes                  # Boxes object for this image
> 
> # Or for a single‑image inference (first element)
> boxes = results[0].boxes
> ```
> 
> **Key points from the code**
> 
> - `Model.predict` returns `self.predictor(source=source, stream=stream)` (source 6).
> - Predictor implementations (e.g., `RTDETRPredictor.postprocess`, `SegmentationPredictor.construct_results`) all return a **list of `Results`** (sources 1, 2, 4).
> - Each `Results` object provides `.boxes`, `.cls`, `.conf`, etc. (source 3 shows `.cls` property of `Boxes`).
> 
> So, after calling the model, you must treat the output as a list and access `.boxes` on each `Results` element.

**Key facts:**

- [X] Calling the model (an alias of Model.predict) returns a list of Results objects, one per image or frame.
- [X] Index or iterate the list to access detections, e.g. results[0].boxes.

## 2. `predict-plot-no-boxes`

**Question:** How can I draw only the segmentation masks on the output image, without the bounding boxes?

> To draw only the segmentation masks (without bounding boxes), use the `Masks` object attached to the `Results` returned by the predictor. The `Masks` class provides the polygon coordinates of each mask in pixel (`xy`) or normalized (`xyn`) form, which you can then render directly.
> 
> **Steps**
> 
> 1. **Get the masks from the result**  
>    ```python
>    # `result` is the object returned by the predictor (e.g., from `predictor.predict()`)
>    masks = result.masks               # type: Masks (see [3])
>    ```
> 
> 2. **Iterate over the mask polygons**  
>    The `xy` property returns a list of `np.ndarray` where each array contains the `[x, y]` pixel coordinates of a mask segment.
> 
> 3. **Draw the masks**  
>    Using OpenCV (or any drawing library) you can fill or outline each polygon:
> 
>    ```python
>    import cv2
> 
>    # `orig_img` is the original image (np.ndarray)
>    for seg in masks.xy:                     # seg shape: (K, 2)
>        # Convert to integer coordinates for OpenCV
>        pts = seg.astype(int).reshape((-1, 1, 2))
>        # Fill the mask with a solid color (e.g., green)
>        cv2.fillPoly(orig_img, [pts], color=(0, 255, 0))
>        # Or draw only the outline:
>        # cv2.polylines(orig_img, [pts], True, color=(0, 255, 0), thickness=2)
>    ```
> 
>    If you prefer normalized coordinates, use `masks.xyn` instead of `masks.xy`.
> 
> **Key points**
> 
> - The `Masks` class is defined in `ult

**Key facts:**

- [ ] Call result.plot(boxes=False): masks are still drawn (masks=True by default) but bounding boxes are not.
- [ ] When saving or showing predictions, predict(..., show_boxes=False) has the same effect.

## 3. `train-lr-schedule`

**Question:** How do the lr0 and lrf hyperparameters work? Is the learning rate decayed linearly or with a cosine schedule?

> **lr0 and lrf**
> 
> - **lr0** – the *initial* learning rate for the optimizer.  
>   Defined in the tuner’s search space (see source [3]):
> 
>   ```python
>   "lr0": (1e-5, 1e-2)   # initial learning rate
>   ```
> 
> - **lrf** – the *final* learning rate expressed as a **fraction of lr0**.  
>   The actual final learning rate is `lr0 * lrf`.  
>   Example: `lr0 = 1e-3`, `lrf = 0.1` → final lr = `1e-4`.  
>   Also defined in the same space (source [3]):
> 
>   ```python
>   "lrf": (0.01, 1.0)    # final learning‑rate fraction (lr0 * lrf)
>   ```
> 
> **Schedule**
> 
> The decay schedule is controlled by the trainer’s `cos_lr` flag (source [2]):
> 
> ```python
> def _setup_scheduler(self):
>     if self.args.cos_lr:
>         self.lf = one_cycle(1, self.args.lrf, self.epochs)   # cosine
>     else:
>         self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf   # linear
>     self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
> ```
> 
> - **If `cos_lr=True`** → a cosine (one‑cycle) schedule from `lr0` down to `lr0 * lrf`.  
> - **If `cos_lr=False`** → a linear schedule from `lr0` down to `lr0 * lrf`.
> 
> Thus, `lr0` sets the starting point, `lrf` sets the end‑point fraction, and the actual decay follows either a cosine or linear curve depending on the `cos_lr` flag.

**Key facts:**

- [x] lr0 is the initial learning rate and lrf the final learning rate as a fraction of lr0 (final lr = lr0 * lrf).
- [x] The decay is linear by default, lf(x) = max(1 - x/epochs, 0) * (1 - lrf) + lrf; a cosine (one-cycle) schedule is used only with cos_lr=True.

## 4. `data-no-labels`

**Question:** Training fails with 'No labels found in labels.cache'. How does YOLO locate the label files for my images and cache them?

> **Short answer:**  
> YOLODataset finds label files with `get_label_files()`, then `cache_labels()` reads them, validates each one via `verify_args()`/`result_to_label()`, and writes a `labels.cache` file. `get_labels()` loads that cache (or rescans if missing/invalid). If the cache contains no labels (`nf == 0`), the training script raises the “No labels found in labels.cache” error.
> 
> ---
> 
> ### Step‑by‑step flow (sources [1‑3])
> 
> | Step | Method | What it does | Key snippet |
> |------|--------|--------------|-------------|
> | **1. Locate labels** | `YOLODataset.get_label_files()` | Returns a list of companion label paths (e.g. `*.txt`) for each image in `self.im_files`. | `def get_label_files(self) -> list[str]: …` |
> | **2. Prepare verification** | `YOLODataset.verify_args()` | Returns a per‑image verification function and its arguments (usually the label file path). | `def verify_args(self) -> tuple: …` |
> | **3. Scan & cache** | `YOLODataset.cache_labels(path)` | <ul><li>Spawns a thread‑pool (`NUM_THREADS`).</li><li>For each image, calls the verification function (from step 2) to read the label file.</li><li>`result_to_label()` converts the raw result into a label dict and updates counters (`nf`, `nm`, `ne`, `nc`).</li><li>If any label dict is produced, it is appended to `x["labels"]`.</li><li>Finally, `x` (labels + counters + hash) is saved to `path` (default `./labels.cache`).</li></ul> | ```python\nwith ThreadPool(NUM_THREADS) as pool:\n    func, iterable = self.verify_args()\n    results = pool.imap(func=func, iterable=iterable)\n    for result in results:\n        label, nm_f, nf_f, ne_f, nc_f, msg = self.result_to_label(result)\n        …\n        if label is not None:\n            x["labels"].append(label)\n``` |
> | **4. Load / rebuild cache** | `YOLODataset._load_or_scan_cache(cache_path, cache_hash)` → called from `get_labels()` | <ul><li>Checks if a cached file exists and its hash matches the current dataset.</li><li>If valid, returns the cached dict; otherwise rescans by calling `cache_labels()`.</li></ul> | `def _load_or_scan_cache(self, cache_path: Path, cache_hash: str) -> tuple[dict, bool]: …` |
> | **5. Return labels for training** | `YOLODataset.get_labels()` | <ul><li>Calls `_load_or_scan_cache` to obtain `labels`.</li><li>If `labels` is empty, raises `RuntimeError("No valid images found …")`.</li><li>Otherwise updates `self.im_files` and calls `self.verify_labels()` before returning.</li></ul> | ```python\nlabels = cache["labels"]\nif not labels:\n    raise RuntimeError(f"No valid images found in {cache_path}.\n  {issues}\n{HELP_URL}")\n``` |
> 
> ---
> 
> ### What triggers the “No labels found in labels.cache” error?
> 
> - In `cache_labels()` (source [2]), after scanning all images, `nf` (number of **found** label files) is checked:
> 
>   ```python
>   if nf == 0:
>       if self.augment:  # training requires labels
>           raise ValueError(f"{self.prefix}No labels found in {path}. {HELP_URL}")
>       LOGGER.warning(...)
>   ```
> 
> - This `ValueError` propagates up to the trainer, which is displayed as the training failure you see.
> 
> **Typical causes**
> 
> 1. **Missing label files** – `get_label_files()` returns paths that don’t exist on disk.
> 2. **Wrong label format** – The verification function (`verify_args`) expects a specific format (e.g., `*.txt` with `class x y w h` lines) and discards malformed files, counting them as `ne` (empty/corrupt).
> 3. **Dataset split mismatch** – If you’re training on a validation split that has no annotations, `nf` will be zero.
> 
> ---
> 
> ### Quick checklist to debug
> 
> ```python
> dataset = YOLODataset(img_path="path/to/images", data={"names": {...}}, task="detect")
> print(dataset.get_label_files()[:5])   # see what paths it expects
> # Verify a few exist:
> import os
> for p in dataset.get_label_files()[:5]:
>     print(p, os.path.isfile(p))
> ```
> 
> If the paths look correct but files are missing, you likely need to generate the label files (e.g., via `yolo dataset` or a data‑preprocessing script) before training.
> 
> ---
> 
> **Bottom line:** YOLO discovers labels automatically via `get_label_files()`, caches them with `cache_labels()` (which validates each file and saves a `labels.cache`), and `get_labels()` either loads that cache or rescans. The error occurs when the cache (or the scan) reports `nf == 0`, i.e., no valid label files were found.

**Key facts:**

- [ ] img2label_paths derives each label path by replacing /images/ with /labels/ in the image path and the extension with .txt.
- [x] Labels are scanned into a labels .cache file that is reused only if its hash of the image and label files matches; otherwise they are rescanned.
- [x] If no valid labels are found, training fails with 'No labels found'.

## 5. `export-tensorrt`

**Question:** How does the TensorRT export work? Does it go through ONNX first?

> **Yes – the TensorRT export goes through an intermediate ONNX file.**
> 
> The `Exporter.export_engine` method (source 1) first calls `self.export_onnx()` to generate an ONNX model, then converts that ONNX file to a TensorRT engine using `onnx2engine`:
> 
> ```python
> def export_engine(self, prefix=colorstr("TensorRT:")):  # noqa: B008
>     """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
>     assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
>     f_onnx = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016
>     from ultralytics.utils.export.engine import onnx2engine
> 
>     assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
>     f = self.file.with_suffix(".engine")  # TensorRT engine file
>     onnx2engine(
>         f_onnx,
>         f,
>         self.args.workspace,
>         self.args.quantize,
>         self.args.dynamic,
>         self.im.shape,
>         dla=self.dla,
>         dataset=self.get_int8_calibration_dataloader(prefix) if self.args.quantize == 8 and not self.qat else None,
>         metadata=self.metadata,
>         verbose=self.args.verbose,
>         prefix=prefix,
>     )
> 
>     return f
> ```
> 
> The `export_onnx` method (source 2) handles the actual ONNX export, including optional simplification, metadata insertion, and FP16/INT8 quantization steps. Only after a valid ONNX file is produced does the TensorRT conversion proceed.

**Key facts:**

- [x] Yes: export_engine first exports an ONNX model with export_onnx and then converts it into a TensorRT .engine file.
- [x] TensorRT export requires a GPU (it fails on CPU).
- [x] Options such as dynamic shapes, workspace size and INT8 quantization (with calibration data) are passed to the conversion.

## 6. `export-openvino`

**Question:** How do I export a YOLO model to OpenVINO, and can the export be quantized to INT8?

> **Export YOLO to OpenVINO (with optional INT8 quantization)**
> 
> You can export a YOLO model to OpenVINO using the `export` method or the CLI. The export supports INT8 quantization when a calibration dataset is provided.
> 
> ```python
> # Python API – export with INT8 quantization
> results = model.export(format='openvino', quantize=8, data='coco8.yaml')
> # results is the path to the exported OpenVINO directory
> ```
> 
> ```bash
> # CLI – export with INT8 quantization
> yolo export model=yolo.pt format=openvino quantize=8 data=coco8.yaml
> ```
> 
> **Key points**
> 
> - The `export_openvino` method (source 1) checks `self.args.quantize`. If `quantize=8`, it builds an NNCF calibration dataset via `self.get_int8_calibration_dataloader` and passes it to `torch2openvino`.
> - The `serialize` function inside `export_openvino` writes the model, metadata, and a `metadata.yaml` file.
> - OpenVINO export supports the `quantize` argument (see `export_formats` in source 5), where `quantize=8` enables INT8 quantization and `quantize=16` enables FP16 compression.
> - INT8 quantization requires a calibration dataset (`data` argument) and the `nncf` package (automatically installed if needed).
> 
> **Example with FP16 (default)**
> 
> ```python
> model.export(format='openvino', quantize=16)   # compresses to FP16
> ```
> 
> The exported model is a directory `*_openvino_model/` containing an `.xml` file (the model) and `metadata.yaml` (model info).

**Key facts:**

- [x] Use model.export(format='openvino'); it writes the OpenVINO .xml/.bin model and metadata.yaml into a *_openvino_model folder.
- [x] INT8 quantization is supported (quantize=8): NNCF calibrates the model with images from the data argument.

## 7. `predict-box-size`

**Question:** How do I get the width and height in pixels of each detected object?

> Based on the source code, to get the width and height in pixels of each detected object, you can access the bounding box coordinates from the results object and calculate the differences.
> 
> Here's how to do it:
> 
> ```python
> # Access the bounding box coordinates
> # For regular detection boxes
> boxes = result.boxes  # Contains xyxy coordinates in pixels
> for i, box in enumerate(boxes):
>     x1, y1, x2, y2 = box.xyxy[0]  # Get first box coordinates
>     width = x2 - x1  # Width in pixels
>     height = y2 - y1  # Height in pixels
>     
> # For OBB (oriented bounding boxes)
> if result.obb is not None:
>     obb = result.obb
>     for i, box in enumerate(obb):
>         # OBB has 4 points, calculate width/height from appropriate points
>         points = box.xyxyxyxy[0]  # 8 coordinates: x1,y1,x2,y2,x3,y3,x4,y4
>         # Calculate width and height based on your specific OBB format
> ```
> 
> The bounding box coordinates (`xyxy`) are stored in pixel values directly from the model output, so subtracting the appropriate coordinates gives you the width and height in pixels.
> 
> For regular detection boxes, the format is `[x1, y1, x2, y2]` where:
> - `x1, y1` = top-left corner
> - `x2, y2` = bottom-right corner
> 
> So:
> - **Width** = `x2 - x1`
> - **Height** = `y2 - y1`
> 
> This approach works for both regular detection and OBB (oriented bounding box) results, though OBB may require additional calculations depending on how you want to define width and height from the 4 corner points.

**Key facts:**

- [ ] Use results[0].boxes.xywh, which gives [x_center, y_center, width, height] for each box.
- [ ] The values are in pixels of the original image, so width = xywh[:, 2] and height = xywh[:, 3] (xywhn is the normalized version).

## 8. `data-mask-convert`

**Question:** My segmentation labels are binary mask images. Can I convert them into YOLO polygon labels?

> Based on the available code, yes! You can convert binary mask images to YOLO polygon labels using the `convert_segment_masks_to_yolo_seg` function.
> 
> Here's how to use it:
> 
> ```python
> from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
> 
> # Convert binary mask images to YOLO polygon labels
> convert_segment_masks_to_yolo_seg(
>     masks_dir="path/to/masks_directory",      # Directory containing your binary mask images
>     output_dir="path/to/output_directory",    # Where to save the converted .txt files
>     classes=80                                 # Total number of classes in your dataset
> )
> ```
> 
> **Key details about the conversion:**
> 
> 1. **Input format**: Your mask images should be grayscale PNG/JPG files where:
>    - Pixel value `0` = background
>    - Pixel value `i+1` = class `i` (e.g., value 1 = class 0, value 2 = class 1, etc.)
> 
> 2. **Output format**: The function creates `.txt` files with YOLO polygon format:
>    - Each line: `class_index x1 y1 x2 y2 ... xn yn`
>    - Coordinates are normalized (0-1) and rounded to 6 decimal places
>    - Each polygon needs at least 3 points
> 
> 3. **Directory structure**:
>    ```
>    masks/
>    ├─ mask_image_01.png
>    ├─ mask_image_02.png
>    └─ ...
>    
>    output_dir/
>    ├─ mask_image_01.txt
>    ├─ mask_image_02.txt
>    └─ ...
>    ```
> 
> The function automatically detects contours from the binary masks and converts them to the required polygon format for YOLO segmentation training. Each connected component in your mask will become a separate polygon in the output.

**Key facts:**

- [x] Yes: convert_segment_masks_to_yolo_seg(masks_dir, output_dir, classes) converts mask images into YOLO segmentation .txt labels.
- [x] Each pixel value v is mapped to class v - 1 (0 is background).
- [x] The contours of each class (cv2.findContours) are written as normalized polygon coordinates.

## 9. `val-test-split`

**Question:** How do I evaluate my trained model on the test split instead of the validation split?

> Based on the indexed code, I can see that the evaluation split is controlled by the `validator.args.split` attribute. Here's how to evaluate on the test split:
> 
> ## Direct Answer
> 
> To evaluate on the test split instead of validation, you need to set the validator's split argument to "test" before calling the validator.
> 
> ## Implementation Options
> 
> ### Option 1: Modify validator args before evaluation
> ```python
> # After training, before final evaluation
> self.validator.args.split = "test"
> self.metrics = self.validator(model=model)
> ```
> 
> ### Option 2: Use a custom validator configuration
> ```python
> # Create a custom validator with test split
> from ultralytics.engine.validator import BaseValidator
> 
> class TestValidator(BaseValidator):
>     def __init__(self, args):
>         super().__init__(args)
>         self.args.split = "test"  # Set test split
> 
> # Then use this validator
> test_validator = TestValidator(args)
> self.metrics = test_validator(model=model)
> ```
> 
> ### Option 3: Pass split as parameter to validator call
> ```python
> # If the validator supports split parameter
> self.metrics = self.validator(model=model, split="test")
> ```
> 
> ## Key Points from the Code
> 
> From the sources, I can see that:
> 
> 1. **WorldTrainerFromScratch.final_eval()** [3] shows how the split is normally set:
>    ```python
>    self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
>    ```
> 
> 2. **BaseTrainer.final_eval()** [6] shows the standard evaluation flow:
>    ```python
>    self.validator.args.plots = self.args.plots
>    self.validator.args.compile = False
>    self.metrics = self.validator(model=model)
>    ```
> 
> 3. **BaseValidator.__call__()** [5] shows that the validator uses `self.args.split` during execution.
> 
> ## Missing Information
> 
> The indexed code doesn't show:
> - Whether the validator's `__call__` method accepts a `split` parameter
> - How to properly configure a validator for test evaluation
> - Whether there are any specific requirements for test set evaluation
> 
> You would need to check the actual validator implementation to see if it supports custom split parameters or if you need to modify the validator's args directly before calling it.

**Key facts:**

- [ ] Call model.val(data='data.yaml', split='test').
- [ ] The validator loads the dataset with check_det_dataset and builds the dataloader from the entry named by split.
- [ ] The dataset YAML must define a test entry; otherwise the images are not found.

## 10. `data-cache`

**Question:** What is the difference between cache='ram' and cache='disk', and how does YOLO check whether there is enough memory?

> **Difference between `cache='ram'` and `cache='disk'`**
> 
> - **`cache='ram'`** – Images are decoded once and kept in a contiguous RAM buffer (`self.ims`). The `load_image` method is used to populate `self.ims[i]`, `self.im_hw0[i]`, `self.im_hw[i]` and the buffer is wrapped in `self._ImageCache` for fast access.  
>   *Source: [4]*
> 
> - **`cache='disk'`** – Images are written to disk as `.npy` files (`self.npy_files[i]`). The `cache_images_to_disk` method writes each image to its own file, and later reads are performed from disk.  
>   *Source: [4]*
> 
> **How YOLO checks whether there is enough memory**
> 
> YOLO calls `BaseDataset.check_cache_ram(self, safety_margin: float = 1.0)` (source [1]) to decide if RAM caching is feasible:
> 
> ```python
> def check_cache_ram(self, safety_margin: float = 1.0) -> bool:
>     b, gb = 0, 1 << 30
>     n = min(self.ni, 30)                     # extrapolate from up to 30 random images
>     for _ in range(n):
>         b += self.load_image(random.randrange(self.ni))[0].nbytes
>     mem_required = b * self.ni / n * (1 + safety_margin)   # bytes needed
>     mem = __import__("psutil").virtual_memory()
>     if mem_required > mem.available:
>         self.cache = None
>         LOGGER.warning(f"{self.prefix}{mem_required / gb:.1f}GB RAM required … but only {mem.available / gb:.1f}GB available")
>         return False
>     return True
> ```
> 
> - It loads a few random images, sums their byte sizes, and extrapolates the total RAM needed for the whole dataset, applying the `safety_margin`.
> - It compares `mem_required` with `psutil.virtual_memory().available`.
> - If insufficient, `self.cache` is set to `None` and a warning is logged; otherwise, caching proceeds.
> 
> Thus, `cache='ram'` stores images in RAM, while `cache='disk'` stores them on disk, and YOLO determines RAM availability via `check_cache_ram` (and disk availability via `check_cache_disk` from source [2]).

**Key facts:**

- [x] cache='ram' (or True) keeps decoded, resized images in memory; cache='disk' saves each image as a .npy file and loads it with np.load.
- [x] check_cache_ram estimates the RAM needed from a sample of images, with a safety margin, and compares it with the available memory.
- [x] If there is not enough memory (or disk space), caching is disabled with a warning.
