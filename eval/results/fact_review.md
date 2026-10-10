# Fact recall review sheet

Blind review of the sources retrieved by `Reranked (nvidia/llama-nemotron-rerank-vl-1b-v2:free)` for 10 questions, sampled at random (seed 0).

For each question, tick `[x]` every key fact that a reader of these sources alone would know: the sources say it or show it directly (e.g. a default value assigned in the code). Leave `[ ]` otherwise. The judge's verdicts are deliberately not shown. When done, run `uv run code-assistant eval agreement --sheet eval/results/fact_review.md --results eval/results/facts.jsonl`.

## 1. `train-lr-schedule`

**Question:** How do the lr0 and lrf hyperparameters work? Is the learning rate decayed linearly or with a cosine schedule?

**Key facts:**

- [x] lr0 is the initial learning rate and lrf the final learning rate as a fraction of lr0 (final lr = lr0 * lrf).
- [x] The decay is linear by default, lf(x) = max(1 - x/epochs, 0) * (1 - lrf) + lrf; a cosine (one-cycle) schedule is used only with cos_lr=True.

<details><summary>Sources</summary>

**[1] `BaseTrainer._setup_scheduler`** (ultralytics/engine/trainer.py, lines 268–274)

```python
def _setup_scheduler(self):
    """Initialize training learning rate scheduler."""
    if self.args.cos_lr:
        self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
    else:
        self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
    self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
```

**[2] `Tuner.__init__`** (ultralytics/engine/tuner.py, lines 79–142)

```python
def __init__(self, args, _callbacks: dict | None = None, *, model=None):
    """Initialize the Tuner with configurations.

    Args:
        args (dict): Configuration for hyperparameter evolution.
        _callbacks (dict | None, optional): Callback functions to be executed during tuning.
        model (torch.nn.Module, optional): Base model whose weights seed each iteration, built from args if None.
    """
    self.space = args.pop("space", None) or {  # key: (min, max, gain(optional))
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": (1e-5, 1e-2),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        "lrf": (0.01, 1.0),  # final learning rate fraction (lr0 * lrf)
        "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
        "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
        "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
        "box": (1.0, 20.0),  # box loss gain
        "cls": (0.1, 4.0),  # cls loss gain (scale with pixels)
        "cls_pw": (0.0, 1.0),  # class weights power
        "dfl": (0.4, 12.0),  # dfl loss gain
        "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": (0.0, 45.0),  # image rotation (+/- deg)
        "translate": (0.0, 0.9),  # image translation (+/- fraction)
        "scale": (0.0, 0.95),  # image scale (+/- gain)
        "shear": (0.0, 10.0),  # image shear (+/- deg)
        "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": (0.0, 1.0),  # image flip up-down (probability)
        "fliplr": (0.0, 1.0),  # image flip left-right (probability)
        "bgr": (0.0, 1.0),  # image channel bgr (probability)
        "mosaic": (0.0, 1.0),  # image mosaic (probability)
        "mixup": (0.0, 1.0),  # image mixup (probability)
        "cutmix": (0.0, 1.0),  # image cutmix (probability)
        "copy_paste": (0.0, 1.0),  # segment/obb copy-paste (object fraction)
        "close_mosaic": (0.0, 10.0),  # close dataloader mosaic (epochs)
    }
    mongodb_uri = args.pop("mongodb_uri", None)
    mongodb_db = args.pop("mongodb_db", "ultralytics")
    mongodb_collection = args.pop("mongodb_collection", "tuner_results")

    self.args = get_cfg(overrides=args)
    if model is None:
        from ultralytics import YOLO

        model = YOLO(self.args.model).model
    self.model = model
    self.args.exist_ok = self.args.resume  # resume w/ same tune_dir
    self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
    self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)  # reset to not affect training
    self.tune_file = self.tune_dir / "tune_results.ndjson"
    self.callbacks = _callbacks or callbacks.get_default_callbacks()
    self.prefix = colorstr("Tuner: ")
    callbacks.add_integration_callbacks(self)

    # MongoDB Atlas support (optional)
    self.mongodb = None
    if mongodb_uri:
        self._init_mongodb(mongodb_uri, mongodb_db, mongodb_collection)

    LOGGER.info(
        f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
        f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
    )
```

**[3] `BaseTrainer._do_train`** (ultralytics/engine/trainer.py, lines 466–542)

```python
def _do_train(self):
    """Perform the full training loop including setup, epoch iteration, validation, and final evaluation."""
    if self.world_size > 1:
        self._setup_ddp()
    self._setup_train()

    nb = len(self.train_loader)  # number of batches
    nw = self._get_warmup_iterations(nb)
    last_opt_step = -1
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    self.run_callbacks("on_train_start")
    LOGGER.info(
        f"Using {len(self.train_loader.dataset)} train, {len(self.test_loader.dataset)} val images for "
        f"fraction={self.args.fraction} at imgsz={self.args.imgsz}\n"
        f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )
    if self.args.close_mosaic:
        base_idx = (self.epochs - self.args.close_mosaic) * nb
        self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = self.start_epoch
    mosaic_closed = not self.args.close_mosaic  # close once when the run enters its final close_mosaic epochs
    self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
    self._oom_retries = 0  # OOM auto-reduce counter for first epoch
    while True:
        self.epoch = epoch
        self.run_callbacks("on_train_epoch_start")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
            self.scheduler.step()

        self._model_train()
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        # Update dataloader attributes (optional)
        if not mosaic_closed and epoch >= self.epochs - self.args.close_mosaic:
            mosaic_closed = True
            self._close_dataloader_mosaic()
            self.train_loader.reset()

        if RANK in {-1, 0}:
            if self.loss_names:
                LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
        self.tloss = None
        for i, batch in pbar:
            self.run_callbacks("on_train_batch_start")
            # Warmup
            ni = i + nb * epoch
            if ni < nw:
                xi = [0, nw]  # x interp
                self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                for x in self.optimizer.param_groups:
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = float(
                        np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                    )
                    if "momentum" in x:
                        x["momentum"] = float(np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum]))

            # Forward
            try:
                with autocast(torch.bfloat16 if self.args.amp == "bf16" else self.amp, device=self.device.type):
                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        # Decouple inference and loss calculations for improved compile performance
```

**[4] `Model.train`** (ultralytics/engine/model.py, lines 807–877)

```python
def train(
    self,
    trainer=None,
    **kwargs: Any,
):
    """Train the model using the specified dataset and training configuration.

    This method facilitates model training with a range of customizable settings. It supports training with a custom
    trainer or the default training approach. The method handles scenarios such as resuming training from a
    checkpoint and updating model and configuration after training. It checks for pip updates and combines default
    configurations, method-specific defaults, and user-provided arguments to configure the training process.

    Args:
        trainer (type[BaseTrainer], optional): Custom trainer class for model training. If None, uses the default
            trainer for the task.
        **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
            - data (str): Path to dataset configuration file.
            - epochs (int): Number of training epochs.
            - batch (int): Batch size for training.
            - imgsz (int): Input image size.
            - device (str): Device to run training on (e.g., 'cuda', 'cpu').
            - workers (int): Number of dataloader worker processes.
            - optimizer (str): Optimizer to use for training.
            - lr0 (float): Initial learning rate.
            - patience (int): Epochs to wait for no observable improvement for early stopping of training.
            - augmentations (list[Callable]): List of augmentation functions to apply during training.

    Returns:
        (ultralytics.utils.metrics.DetMetrics | dict | None): Training metrics if available and training is
            successful; otherwise, None. The specific metrics type depends on the task. When `data` is a list or
            tuple of datasets, the base model is fine-tuned on each in series and a {dataset: metrics} dict is
            returned.

    Raises:
        TypeError: If the model is not a PyTorch model.
        FileNotFoundError: If training completes but no checkpoint was saved.

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.train(data="coco8.yaml", epochs=3)
        >>> multi = model.train(data=["coco8.yaml", "african-wildlife.yaml"], epochs=3)  # fine-tune across datasets
    """
    self._check_is_pytorch_model()
    checks.check_pip_update_available()

    overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
    custom = {
        # NOTE: handle the case when 'cfg' includes 'data'.
        "data": (overrides.get("data") if kwargs.get("cfg") else None)
        or DEFAULT_CFG_DICT["data"]
        or (
            None if isinstance(kwargs.get("resume", overrides.get("resume")), (str, Path)) else TASK2DATA[self.task]
        ),
        "model": self.overrides["model"],
        "task": self.task,
    }  # method defaults
    args = {**overrides, **custom, **kwargs, "mode": "train"}  # prioritizes rightmost args
    pretrained = args.setdefault("pretrained", self.overrides.get("pretrained", True))
    if isinstance(args.get("data"), (list, tuple)):  # fine-tune a single base model across multiple datasets
        from ultralytics.engine.trainer import MultiTrainer

        use_python_trainer = trainer is not None or self.callbacks != callbacks.get_default_callbacks()
        self.trainer = MultiTrainer(
            (trainer or self._smart_load("trainer")) if use_python_trainer else None,
            args,
            self.model,
            _callbacks=self.callbacks,
        )
        self.metrics = self.trainer.train()
        return self.metrics
    if args.get("resume") is True:  # resume=True (boolean) uses current model as checkpoint
```

**[5] `TinyViT.set_layer_lr_decay`** (ultralytics/models/sam/modules/tiny_encoder.py, lines 900–933)

```python
def set_layer_lr_decay(self, layer_lr_decay: float):
    """Set layer-wise learning rate decay for the TinyViT model based on depth."""
    decay_rate = layer_lr_decay

    # Layers -> blocks (depth)
    depth = sum(self.depths)
    lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

    def _set_lr_scale(m, scale):
        """Set the learning rate scale for each layer in the model based on the layer's depth."""
        for p in m.parameters():
            p.lr_scale = scale

    self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
    i = 0
    for layer in self.layers:
        for block in layer.blocks:
            block.apply(lambda x, scale=lr_scales[i]: _set_lr_scale(x, scale))
            i += 1
        if layer.downsample is not None:
            layer.downsample.apply(lambda x, scale=lr_scales[i - 1]: _set_lr_scale(x, scale))
    assert i == depth
    for m in {self.norm_head, self.head}:
        m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

    for k, p in self.named_parameters():
        p.param_name = k

    def _check_lr_scale(m):
        """Check if the learning rate scale attribute is present in module's parameters."""
        for p in m.parameters():
            assert hasattr(p, "lr_scale"), p.param_name

    self.apply(_check_lr_scale)
```

**[6] `TinyViT.__init__`** (ultralytics/models/sam/modules/tiny_encoder.py, lines 787–864)

```python
def __init__(
    self,
    img_size: int = 224,
    in_chans: int = 3,
    num_classes: int = 1000,
    embed_dims: tuple[int, int, int, int] = (96, 192, 384, 768),
    depths: tuple[int, int, int, int] = (2, 2, 6, 2),
    num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
    window_sizes: tuple[int, int, int, int] = (7, 7, 14, 7),
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    use_checkpoint: bool = False,
    mbconv_expand_ratio: float = 4.0,
    local_conv_size: int = 3,
    layer_lr_decay: float = 1.0,
):
    """Initialize the TinyViT model.

    This constructor sets up the TinyViT architecture, including patch embedding, multiple layers of attention and
    convolution blocks, and a classification head.

    Args:
        img_size (int, optional): Size of the input image.
        in_chans (int, optional): Number of input channels.
        num_classes (int, optional): Number of classes for classification.
        embed_dims (tuple[int, int, int, int], optional): Embedding dimensions for each stage.
        depths (tuple[int, int, int, int], optional): Number of blocks in each stage.
        num_heads (tuple[int, int, int, int], optional): Number of attention heads in each stage.
        window_sizes (tuple[int, int, int, int], optional): Window sizes for each stage.
        mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim.
        drop_rate (float, optional): Dropout rate.
        drop_path_rate (float, optional): Stochastic depth rate.
        use_checkpoint (bool, optional): Whether to use checkpointing to save memory.
        mbconv_expand_ratio (float, optional): Expansion ratio for MBConv layer.
        local_conv_size (int, optional): Kernel size for local convolutions.
        layer_lr_decay (float, optional): Layer-wise learning rate decay factor.
    """
    super().__init__()
    self.img_size = img_size
    self.num_classes = num_classes
    self.depths = depths
    self.num_layers = len(depths)
    self.mlp_ratio = mlp_ratio

    activation = nn.GELU

    self.patch_embed = PatchEmbed(
        in_chans=in_chans, embed_dim=embed_dims[0], resolution=img_size, activation=activation
    )

    patches_resolution = self.patch_embed.patches_resolution
    self.patches_resolution = patches_resolution

    # Stochastic depth
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

    # Build layers
    self.layers = nn.ModuleList()
    for i_layer in range(self.num_layers):
        kwargs = {
            "dim": embed_dims[i_layer],
            "input_resolution": (
                patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
            ),
            #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
            #                     patches_resolution[1] // (2 ** i_layer)),
            "depth": depths[i_layer],
            "drop_path": dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            "downsample": PatchMerging if (i_layer < self.num_layers - 1) else None,
            "use_checkpoint": use_checkpoint,
            "out_dim": embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
            "activation": activation,
        }
        if i_layer == 0:
            layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
        else:
```

</details>

## 2. `train-tune`

**Question:** How does model.tune() search for better hyperparameters? Is it a grid search?

**Key facts:**

- [x] It is not a grid search: Tuner evolves hyperparameters by mutating the best previous results with Gaussian noise (evolutionary search).
- [x] Mutations are clipped to a search space of (min, max) bounds for hyperparameters such as lr0, lrf, momentum and augmentation settings.
- [x] Each iteration trains with the mutated hyperparameters and records its fitness; the best set is saved to best_hyperparameters.yaml.

<details><summary>Sources</summary>

**[1] `Model.tune`** (ultralytics/engine/model.py, lines 915–966)

```python
def tune(
    self,
    use_ray=False,
    iterations=300,
    *args: Any,
    **kwargs: Any,
):
    """Conduct hyperparameter tuning for the model, with an option to use Ray Tune.

    This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method. When Ray Tune
    is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module. Otherwise, it uses
    the internal 'Tuner' class for tuning. The method combines default, overridden, and custom arguments to
    configure the tuning process.

    Args:
        use_ray (bool): Whether to use Ray Tune for hyperparameter tuning. If False, uses internal tuning method.
        iterations (int): Number of tuning iterations to perform.
        *args (Any): Positional arguments forwarded to Ray Tune's `run_ray_tune` (space, grace_period,
            gpu_per_trial) when `use_ray=True`; the internal Tuner raises a TypeError if any are given.
        **kwargs (Any): Additional keyword arguments for tuning configuration. These are combined with model
            overrides and defaults to configure the tuning process.

    Returns:
        (ray.tune.ResultGrid | None): When use_ray=True, returns a ResultGrid with hyperparameter search results.
            When use_ray=False, returns None and saves best hyperparameters to YAML.

    Raises:
        TypeError: If the model is not a PyTorch model, or positional args are passed with use_ray=False.

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.tune(data="coco8.yaml", iterations=5)
        >>> print(results)

        # Use Ray Tune for more advanced hyperparameter search
        >>> results = model.tune(use_ray=True, iterations=20, data="coco8.yaml")
    """
    self._check_is_pytorch_model()
    if "optimizer" not in (kwargs.get("space") or {}):
        kwargs.setdefault("optimizer", "AdamW")
    if use_ray:
        from ultralytics.utils.tuner import run_ray_tune

        return run_ray_tune(self, *args, iterations=iterations, **kwargs)
    else:
        if args:
            raise TypeError(f"Positional arguments {args} require use_ray=True, pass Tuner options as keywords.")
        from .tuner import Tuner

        custom = {}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        return Tuner(args=args, model=self.model, _callbacks=self.callbacks)(iterations=iterations)
```

**[2] `ultralytics.engine.tuner`** (ultralytics/engine/tuner.py, lines 1–619)

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Hyperparameter tuning for Ultralytics YOLO models across all supported tasks.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Examples:
    Tune hyperparameters for YOLO26n on COCO8 at imgsz=640 and epochs=10 for 300 tuning iterations.
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolo26n.pt")
    >>> model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
"""
from __future__ import annotations
import json
import shutil
import time
from datetime import datetime
import numpy as np
from ultralytics.cfg import CFG_INT_KEYS, TASK2METRIC, get_cfg, get_save_dir
from ultralytics.utils import LOGGER, YAML, callbacks, colorstr, remove_colorstr
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.plotting import plot_tune_results
class Tuner():
    """A class for hyperparameter tuning of YOLO models."""
```

**[3] `Tuner`** (ultralytics/engine/tuner.py, lines 30–619)

```python
    @staticmethod
    def _best_metrics(result: dict) -> dict | None:
        """Summarize best-result metrics for logging."""
        ...
    @staticmethod
    def _has_training_metrics(result: dict, require_all: bool=False) -> bool:
        """Return whether a tuning result contains training metrics."""
        ...
    @classmethod
    def _best_result_index(cls, results: list[dict], fitness: np.ndarray) -> int:
        """Return the best result index, preferring rows with training metrics."""
        ...
    def _mutate(self, n: int=9, sigma: float=0.2) -> dict[str, float]:
        """Mutate hyperparameters based on bounds and scaling factors specified in `self.space`."""
        ...
    def _constrain(self, hyp: dict[str, float]) -> dict[str, float]:
        """Constrain hyperparameters to their search bounds and configured types."""
        ...
    def __call__(self, iterations: int=300, cleanup: bool=True):
        """Execute the hyperparameter evolution process when the Tuner instance is called."""
        ...
```

**[4] `Tuner.__init__`** (ultralytics/engine/tuner.py, lines 79–142)

```python
def __init__(self, args, _callbacks: dict | None = None, *, model=None):
    """Initialize the Tuner with configurations.

    Args:
        args (dict): Configuration for hyperparameter evolution.
        _callbacks (dict | None, optional): Callback functions to be executed during tuning.
        model (torch.nn.Module, optional): Base model whose weights seed each iteration, built from args if None.
    """
    self.space = args.pop("space", None) or {  # key: (min, max, gain(optional))
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": (1e-5, 1e-2),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        "lrf": (0.01, 1.0),  # final learning rate fraction (lr0 * lrf)
        "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
        "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
        "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
        "box": (1.0, 20.0),  # box loss gain
        "cls": (0.1, 4.0),  # cls loss gain (scale with pixels)
        "cls_pw": (0.0, 1.0),  # class weights power
        "dfl": (0.4, 12.0),  # dfl loss gain
        "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": (0.0, 45.0),  # image rotation (+/- deg)
        "translate": (0.0, 0.9),  # image translation (+/- fraction)
        "scale": (0.0, 0.95),  # image scale (+/- gain)
        "shear": (0.0, 10.0),  # image shear (+/- deg)
        "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": (0.0, 1.0),  # image flip up-down (probability)
        "fliplr": (0.0, 1.0),  # image flip left-right (probability)
        "bgr": (0.0, 1.0),  # image channel bgr (probability)
        "mosaic": (0.0, 1.0),  # image mosaic (probability)
        "mixup": (0.0, 1.0),  # image mixup (probability)
        "cutmix": (0.0, 1.0),  # image cutmix (probability)
        "copy_paste": (0.0, 1.0),  # segment/obb copy-paste (object fraction)
        "close_mosaic": (0.0, 10.0),  # close dataloader mosaic (epochs)
    }
    mongodb_uri = args.pop("mongodb_uri", None)
    mongodb_db = args.pop("mongodb_db", "ultralytics")
    mongodb_collection = args.pop("mongodb_collection", "tuner_results")

    self.args = get_cfg(overrides=args)
    if model is None:
        from ultralytics import YOLO

        model = YOLO(self.args.model).model
    self.model = model
    self.args.exist_ok = self.args.resume  # resume w/ same tune_dir
    self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
    self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)  # reset to not affect training
    self.tune_file = self.tune_dir / "tune_results.ndjson"
    self.callbacks = _callbacks or callbacks.get_default_callbacks()
    self.prefix = colorstr("Tuner: ")
    callbacks.add_integration_callbacks(self)

    # MongoDB Atlas support (optional)
    self.mongodb = None
    if mongodb_uri:
        self._init_mongodb(mongodb_uri, mongodb_db, mongodb_collection)

    LOGGER.info(
        f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
        f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
    )
```

**[5] `Tuner.__call__`** (ultralytics/engine/tuner.py, lines 473–551)

```python
def __call__(self, iterations: int = 300, cleanup: bool = True):
    """Execute the hyperparameter evolution process when the Tuner instance is called.

    This method iterates through the specified number of iterations, performing the following steps:
    1. Sync MongoDB results to local NDJSON (if using distributed mode)
    2. Mutate hyperparameters using the best previous results or defaults
    3. Train a YOLO model with the mutated hyperparameters
    4. Log fitness scores and hyperparameters to MongoDB and/or NDJSON
    5. Track the best performing configuration across all iterations

    Args:
        iterations (int): The number of generations to run the evolution for.
        cleanup (bool): Whether to delete non-best iteration run directories to reduce storage space during tuning.
    """
    from ultralytics.engine.trainer import MultiTrainer

    t0 = time.time()
    self.tune_dir.mkdir(parents=True, exist_ok=True)
    (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)
    best_save_dirs = {}
    n_successful = 0  # iters with real training metrics in this invocation (excludes resumed/MongoDB rows)

    # Sync MongoDB to local NDJSON at startup for proper resume logic
    if self.mongodb:
        self._sync_mongodb_to_file()

    start = 0
    if self.tune_file.exists():
        start = len(self._load_local_results())
        LOGGER.info(f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}...")
    for i in range(start, iterations):
        # Mutate hyperparameters
        mutated_hyp = self._mutate()
        LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")

        train_args = {**vars(self.args), **mutated_hyp}
        data = train_args.pop("data")
        if not isinstance(data, (list, tuple)):
            data = [data]
        trainer = MultiTrainer(None, {**train_args, "data": data}, self.model)
        raw_metrics = trainer.train()
        failed_datasets = [dataset for dataset, metrics in raw_metrics.items() if not metrics]
        metric = TASK2METRIC[train_args["task"]]
        dataset_metrics = {
            dataset: metrics or {metric: 0.0, "fitness": 0.0} for dataset, metrics in raw_metrics.items()
        }
        save_dir = [trainer.save_dir / dataset for dataset in dataset_metrics]
        weights_dir = [s / "weights" for s in save_dir]
        metrics = trainer.mean_metrics
        fitness = sum((metrics or {}).get("fitness") or 0.0 for metrics in dataset_metrics.values()) / len(data)
        metrics["fitness"] = fitness
        result = self._result_record(
            i + 1,
            fitness,
            mutated_hyp,
            dataset_metrics,
            {dataset: str(s) for dataset, s in zip(dataset_metrics, save_dir)},
            failed_datasets,
        )
        if self._has_training_metrics(result, require_all=True):
            n_successful += 1
        stop_after_iteration = False
        if self.mongodb:
            self._save_to_mongodb(
                fitness, mutated_hyp, metrics, dataset_metrics, result["save_dirs"], failed_datasets
            )
            self._sync_mongodb_to_file()
            total_mongo_iterations = self.collection.count_documents({"fitness": {"$exists": True}})
            if total_mongo_iterations >= iterations:
                stop_after_iteration = True
        else:
            self._save_local_result(result)

        # Get best results
        results = self._load_local_results()
        x = self._local_results_to_array(results)
        fitness = x[:, 0]  # first column
        best_idx = self._best_result_index(results, fitness)
        best_result = results[best_idx]
```

**[6] `Tuner._mutate`** (ultralytics/engine/tuner.py, lines 436–458)

```python
            )
        for attempt in range(200):
            if attempt < 100:
                genes = population[rng.choice(len(x), p=weights / weights.sum())]
                mask = (rng.random(ng) < 0.5) & mutable
                if covariance is not None and rng.random() < 0.4 * confidence:
                    genes = np.where(mask, rng.multivariate_normal(genes, covariance), genes)
                    genes = 1 - np.abs(genes % 2 - 1)
                else:
                    genes = np.clip(genes + mask * rng.standard_normal(ng) * scale, 0, 1)
            else:
                genes = population[rng.choice(len(x), p=weights / weights.sum())].copy()
                genes[mutable] = rng.random(mutable.sum())
            hyp = {k: float(bounds[i, 0] + genes[i] * span[i]) for i, k in enumerate(self.space)}
            hyp = self._constrain(hyp)
            if tuple(hyp.values()) not in existing:
                return hyp
        raise RuntimeError(f"{self.prefix}Unable to generate a unique hyperparameter mutation")
    raise RuntimeError(f"{self.prefix}Hyperparameter search space is exhausted")
else:
    hyp = {k: getattr(self.args, k) for k in self.space}

return self._constrain(hyp)
```

</details>

## 3. `data-no-labels`

**Question:** Training fails with 'No labels found in labels.cache'. How does YOLO locate the label files for my images and cache them?

**Key facts:**

- [ ] img2label_paths derives each label path by replacing /images/ with /labels/ in the image path and the extension with .txt.
- [x] Labels are scanned into a labels .cache file that is reused only if its hash of the image and label files matches; otherwise they are rescanned.
- [x] If no valid labels are found, training fails with 'No labels found'.

<details><summary>Sources</summary>

**[1] `YOLODataset.cache_labels`** (ultralytics/data/dataset.py, lines 120–165)

```python
def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
    """Cache dataset labels, check images and read shapes.

    This is the shared scanning skeleton for file-based datasets; subclasses customize it through the
    `get_label_files`, `get_cache_hash`, `verify_args`, `result_to_label` and `scan_summary` hooks instead of
    duplicating this method.

    Args:
        path (Path): Path where to save the cache file.

    Returns:
        (dict): Dictionary containing cached labels and related information.
    """
    x = {"labels": []}
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
    total = len(self.im_files)
    with ThreadPool(NUM_THREADS) as pool:
        func, iterable = self.verify_args()
        results = pool.imap(func=func, iterable=iterable)
        pbar = TQDM(results, desc=desc, total=total)
        for result in pbar:
            label, nm_f, nf_f, ne_f, nc_f, msg = self.result_to_label(result)
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if label is not None:
                x["labels"].append(label)
            if msg:
                msgs.append(msg)
            pbar.desc = f"{desc} {self.scan_summary(nf, nm, ne, nc)}"
        pbar.close()

    if msgs:
        LOGGER.info("\n".join(msgs))
    if nf == 0:
        if self.augment:  # training requires labels; unlabeled val splits (e.g. COCO test-dev) only warn
            raise ValueError(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
    x["hash"] = self.get_cache_hash()
    x["results"] = nf, nm, ne, nc, total
    x["msgs"] = msgs  # warnings
    if x["labels"]:
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
    return x
```

**[2] `YOLODataset.get_labels`** (ultralytics/data/dataset.py, lines 291–322)

```python
def get_labels(self) -> list[dict]:
    """Return list of label dictionaries for YOLO training.

    This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

    Returns:
        (list[dict]): List of label dictionaries, each containing information about an image and its annotations.

    Raises:
        RuntimeError: If no valid images are found.
    """
    label_files = self.get_label_files()
    cache_path = Path(label_files[0]).parent.with_suffix(".cache")
    cache, exists = self._load_or_scan_cache(cache_path, self.get_cache_hash())

    # Display cache
    nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
    if exists and LOCAL_RANK in {-1, 0}:
        d = f"Scanning {cache_path}... {self.scan_summary(nf, nm, ne, nc)}"
        TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
        if cache["msgs"]:
            LOGGER.info("\n".join(cache["msgs"]))  # display warnings

    # Read cache
    labels = cache["labels"]
    if not labels:
        issues = "\n  ".join(sorted(set(cache["msgs"]))) or "no error details"
        raise RuntimeError(f"No valid images found in {cache_path}.\n  {issues}\n{HELP_URL}")
    [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
    self.im_files = [lb["im_file"] for lb in labels]  # update im_files
    self.verify_labels(labels, cache_path)
    return labels
```

**[3] `YOLODataset`** (ultralytics/data/dataset.py, lines 61–458)

```python
class YOLODataset(BaseDataset):
    """Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, instance segmentation, pose estimation, and oriented bounding
    box (OBB) tasks using the YOLO format.

    Attributes:
        format_class (type[Format]): Formatter appended by build_transforms; subclasses override it per task.
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Return list of label dictionaries for YOLO training.
        get_label_files: Return companion label files for the dataset's images.
        verify_args: Return the per-image verification function and its arguments.
        result_to_label: Convert one verification result into a label dict.
        verify_labels: Check box/segment consistency of the loaded labels.
        get_cache_hash: Return the hash used to validate a label cache.
        scan_summary: Return a one-line summary of scan counters.
        build_transforms: Build and append transforms to the list.
        build_text_transforms: Insert text augmentation for text-based subclasses.
        close_mosaic: Disable mosaic, copy_paste, mixup and cutmix augmentations and build transformations.
        update_labels_info: Update label format for different tasks.
        collate_fn: Collate data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """
    format_class = Format
    def __init__(self, *args, data: dict, task: str='detect', **kwargs):
        """Initialize the YOLODataset."""
        ...
    def cache_labels(self, path: Path=Path('./labels.cache')) -> dict:
        """Cache dataset labels, check images and read shapes."""
        ...
    def get_label_files(self) -> list[str]:
        """Return the companion label files for the dataset's images, storing them on the instance."""
        ...
    def get_cache_hash(self) -> str:
        """Return the hash used to validate a label cache against the current dataset files and scan settings."""
        ...
    def scan_summary(self, nf: int, nm: int, ne: int, nc: int) -> str:
        """Return a one-line summary of scan counters for progress bars and cache logs."""
        ...
    def verify_args(self) -> tuple:
        """Return the per-image verification function and its argument iterable used by `cache_labels`."""
        ...
    def result_to_label(self, result: list) -> tuple[dict | None, int, int, int, int, str]:
        """Convert one verification result into a label dict and scan counter increments."""
        ...
    def verify_labels(self, labels: list[dict], cache_path: Path) -> None:
        """Check that the dataset is all boxes or all segments, removing mixed segments if necessary."""
        ...
    def _load_or_scan_cache(self, cache_path: Path, cache_hash: str) -> tuple[dict, bool]:
        """Load a dataset cache file if it matches the current version and hash, otherwise rescan and rebuild it."""
        ...
    def get_labels(self) -> list[dict]:
        """Return list of label dictionaries for YOLO training."""
        ...
    def build_transforms(self, hyp: IterableSimpleNamespace) -> Compose:
        """Build and append transforms to the list."""
        ...
    def build_text_transforms(self, transforms: Compose, max_samples: int) -> Compose:
        """Insert text augmentation for text-based subclasses providing `category_freq`."""
        ...
```

**[4] `YOLODataset.get_label_files`** (ultralytics/data/dataset.py, lines 167–174)

```python
def get_label_files(self) -> list[str]:
    """Return the companion label files for the dataset's images, storing them on the instance.

    Returns:
        (list[str]): List of label file paths.
    """
    self.label_files = img2label_paths(self.im_files)
    return self.label_files
```

**[5] `GroundingDataset.cache_labels`** (ultralytics/data/dataset.py, lines 691–773)

```python
def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
    """Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

    Args:
        path (Path): Path where to save the cache file.

    Returns:
        (dict[str, Any]): Dictionary containing cached labels and related information.
    """
    x = {"labels": []}
    LOGGER.info("Loading annotation file...")
    with open(self.json_file) as f:
        annotations = json.load(f)
    images = {f"{x['id']:d}": x for x in annotations["images"]}
    img_to_anns = defaultdict(list)
    for ann in annotations["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
    dropped = False
    for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
        img = images[f"{img_id:d}"]
        h, w, f = img["height"], img["width"], img["file_name"]
        im_file = Path(self.img_path) / f
        if not im_file.exists():
            continue
        bboxes = []
        segments = []
        segmented = False
        cat2id = {}
        texts = []
        for ann in anns:
            if ann["iscrowd"]:
                continue
            box = np.array(ann["bbox"], dtype=np.float32)
            box[:2] += box[2:] / 2
            box[[0, 2]] /= float(w)
            box[[1, 3]] /= float(h)
            if box[2] <= 0 or box[3] <= 0:
                continue

            caption = img["caption"]
            cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
            if not cat_name:
                continue

            if cat_name not in cat2id:
                cat2id[cat_name] = len(cat2id)
                texts.append([cat_name])
            cls = cat2id[cat_name]  # class
            box = [cls, *box.tolist()]
            if box not in bboxes:
                bboxes.append(box)
                raw_seg = ann.get("segmentation")
                segmented |= raw_seg is not None
                seg = raw_seg if isinstance(raw_seg, list) else []
                polygons = [
                    p
                    for p in seg
                    if isinstance(p, list)
                    and len(p) >= 6
                    and not len(p) % 2
                    and all(isinstance(c, (int, float)) for c in p)
                ]
                dropped |= bool(raw_seg) and (not isinstance(raw_seg, list) or len(polygons) < len(seg))
                if not polygons:  # keep one segment per box so an image mixing the two kinds stays aligned
                    cx, cy, bw, bh = box[1:]
                    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                    segments.append([cls, x1, y1, x2, y1, x2, y2, x1, y2])  # segments2boxes returns the box
                    continue
                elif len(polygons) > 1:
                    s = merge_multi_segment(polygons)
                    s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                else:
                    s = [j for i in polygons for j in i]  # all segments concatenated
                    s = (
                        (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                        .reshape(-1)
                        .tolist()
                    )
                segments.append([cls, *s])
        lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

        if segmented:
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
```

**[6] `YOLODataset.get_cache_hash`** (ultralytics/data/dataset.py, lines 176–185)

```python
def get_cache_hash(self) -> str:
    """Return the hash used to validate a label cache against the current dataset files and scan settings.

    Returns:
        (str): Dataset cache hash.
    """
    # add_polygon_background() class is not a label class, so segment and semantic share one cache
    nc = self.data.get("bg_class_idx") or len(self.data["names"])
    scan_args = (self.use_keypoints, nc, self.data.get("kpt_shape"), self.single_cls)
    return get_hash(self.label_files + self.im_files + [str(scan_args)])
```

</details>

## 4. `data-mask-convert`

**Question:** My segmentation labels are binary mask images. Can I convert them into YOLO polygon labels?

**Key facts:**

- [x] Yes: convert_segment_masks_to_yolo_seg(masks_dir, output_dir, classes) converts mask images into YOLO segmentation .txt labels.
- [x] Each pixel value v is mapped to class v - 1 (0 is background).
- [x] The contours of each class (cv2.findContours) are written as normalized polygon coordinates.

<details><summary>Sources</summary>

**[1] `PolygonSemanticDataset`** (ultralytics/data/dataset.py, lines 1105–1158)

```python
class PolygonSemanticDataset(SemanticDataset, YOLODataset):
    """Semantic segmentation dataset that rasterizes YOLO polygon labels into masks on the fly.

    Used when the dataset YAML lacks 'masks_dir'. Pixels not covered by any polygon become a dedicated background class.
    Requires `add_polygon_background(data)` to be called first: for nc > 1 it bumps `data['nc']` to user_nc + 1 with
    background at `nc - 1`; for nc == 1 it keeps nc=1 and rasterizes a {0=bg, 1=fg} binary mask for use with
    BCEWithLogitsLoss.
    """
    def __init__(self, *args, data: dict, **kwargs):
        """Initialize PolygonSemanticDataset."""
        ...
    # Rebind label scanning to YOLODataset's polygon .txt implementations; the MRO (SemanticDataset, YOLODataset)
    # would otherwise resolve SemanticDataset's PNG-mask hooks and its get_labels, which syncs mask_files from
    # label dicts that polygon labels do not have.
    get_labels = YOLODataset.get_labels
    get_label_files = YOLODataset.get_label_files
    get_cache_hash = YOLODataset.get_cache_hash
    scan_summary = YOLODataset.scan_summary
    verify_args = YOLODataset.verify_args
    result_to_label = YOLODataset.result_to_label
    verify_labels = YOLODataset.verify_labels
    def load_mask(self, index: int, image_shape: tuple[int, int] | None=None) -> np.ndarray:
        """Rasterize this image's polygons into a (H, W) uint8 semantic mask, bg = self.bg_class_idx."""
        ...
```

**[2] `convert_segment_masks_to_yolo_seg`** (ultralytics/data/converter.py, lines 377–452)

```python
def convert_segment_masks_to_yolo_seg(masks_dir: str, output_dir: str, classes: int):
    """Convert a dataset of segmentation mask images to the YOLO segmentation format.

    This function takes the directory containing grayscale mask images, where each pixel value is the class index + 1
    and 0 is background, and converts them into YOLO segmentation format. The converted labels are saved in the
    specified output directory with the same file stems as the masks.

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg, jpeg) are stored.
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        classes (int): Total number of classes in the dataset, e.g., 80 for COCO.

    Examples:
        >>> from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
        >>> convert_segment_masks_to_yolo_seg("path/to/masks_directory", "path/to/output/directory", classes=80)

    Notes:
        The expected directory structure for the masks is:

            - masks
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        After execution, the labels will be organized in the following structure:

            - output_dir
                ├─ mask_image_01.txt
                ├─ mask_image_02.txt
                ├─ mask_image_03.txt
                └─ mask_image_04.txt
    """
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in sorted(Path(masks_dir).iterdir()):
        if mask_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(mask_path) as im:
                mask = read_mask(str(mask_path), im.mode)
            img_height, img_width = mask.shape  # Get image dimensions
            LOGGER.info(f"Processing {mask_path} imgsz = {img_height} x {img_width}")

            unique_values = np.unique(mask)  # Get unique pixel values representing different classes
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # Skip background
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    LOGGER.warning(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # Create a binary mask for the current class and find contours
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # Find contours

                for contour in contours:
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        yolo_format = [class_index]
                        for point in contour:
                            # Normalize the coordinates
                            yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # Save Ultralytics YOLO format data to file
            output_path = output_dir / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
```

**[3] `yolo_bbox2segment`** (ultralytics/data/converter.py, lines 614–676)

```python
def yolo_bbox2segment(
    im_dir: str | Path, save_dir: str | Path | None = None, sam_model: str = "sam_b.pt", device: int | str | None = None
):
    """Convert existing object detection dataset (bounding boxes) to segmentation dataset in YOLO format.

    Generates segmentation data using SAM auto-annotator as needed.

    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path, optional): Path to save the generated labels, labels will be saved into `labels-segment`
            in the same directory level of `im_dir` if save_dir is None.
        sam_model (str): Segmentation model to use for intermediate segmentation data.
        device (int | str, optional): The specific device to run SAM models.

    Notes:
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ...
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ...
                └─ NNN.txt
    """
    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils.ops import xywh2xyxy

    # NOTE: add placeholder to pass class index check
    dataset = YOLODataset(im_dir, data={"names": list(range(1000)), "channels": 3})
    if any(len(lb["segments"]) for lb in dataset.labels):  # segment data, any label since background images have none
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model)
    for label in TQDM(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):
        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:  # skip empty labels
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        sam_results = sam_model(label["im_file"], bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)
        label["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = label["cls"]
        for i, s in enumerate(label["segments"]):
            if len(s) < 3:  # fewer than 3 points is not a polygon, and writes a row no loader accepts
                continue
            line = (int(cls[i, 0]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.writelines(text + "\n" for text in texts)
    LOGGER.info(f"Generated segment labels saved in {save_dir}")
```

**[4] `ultralytics.data.utils`** (ultralytics/data/utils.py, lines 1–1046)

```python
def polygon2mask(imgsz: tuple[int, int], polygons: list[np.ndarray], color: int=1, downsample_ratio: int=1) -> np.ndarray:
    """Convert a list of polygons to a binary mask of the specified image size."""
    ...
def polygons2masks(imgsz: tuple[int, int], polygons: list[np.ndarray], color: int, downsample_ratio: int=1) -> np.ndarray:
    """Convert a list of polygons to a set of binary masks of the specified image size."""
    ...
def polygons2masks_overlap(imgsz: tuple[int, int], segments: list[np.ndarray], downsample_ratio: int=1) -> tuple[np.ndarray, np.ndarray]:
    """Return a downsampled overlap mask and sorted area indices."""
    ...
def find_dataset_yaml(path: Path) -> Path:
    """Find and return the YAML file associated with a Detect, Segment or Pose dataset."""
    ...
def get_split_fraction(fraction: float | list[float | int], split: str) -> float | int:
    """Return a split ratio/count, normalizing boundary values to 0.0 (none) or 1.0 (all)."""
    ...
def convert_ndjson_to_yolo_if_needed(data: str | Path, fraction: float | list[float | int]=1.0, *, split: str | None=None) -> str | Path:
    """Convert an NDJSON dataset or Platform dataset URI to YOLO format."""
    ...
def check_det_dataset(dataset: str | Path, autodownload: bool=True, split: str='') -> dict[str, Any]:
    """Download, verify, and/or unzip a dataset if not found locally."""
    ...
def check_cls_dataset(dataset: str | Path, split: str='') -> dict[str, Any]:
    """Check a classification dataset such as Imagenet."""
    ...
def compress_one_image(f: str | Path, f_new: str | Path | None=None, max_dim: int=1920, quality: int=50):
    """Compress a single image file to reduced size while preserving its aspect ratio."""
    ...
def load_dataset_cache_file(path: Path) -> dict:
    """Load an Ultralytics *.cache dictionary from path."""
    ...
def save_dataset_cache_file(prefix: str, path: Path, x: dict, version: str):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    ...
def add_polygon_background(data: dict) -> dict:
    """Set up the background class for polygon-based semantic datasets without 'masks_dir'."""
    ...
```

**[5] `GroundingDataset.cache_labels`** (ultralytics/data/dataset.py, lines 771–797)

```python

    if segmented:
        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
        lb[:, 1:] = segments2boxes(segments)  # boxes follow the polygons
    else:
        segments = []  # no annotation carried a segmentation, so store no masks

    x["labels"].append(
        {
            "im_file": im_file,
            "shape": (h, w),
            "cls": lb[:, 0:1],  # n, 1
            "bboxes": lb[:, 1:],  # n, 4
            "segments": segments,
            "normalized": True,
            "bbox_format": "xywh",
            "texts": texts,
        }
    )
if dropped:
    LOGGER.warning(
        f"{self.json_file}: ignored segmentations that are not polygon point lists, such as RLE masks. "
        "Annotations left without a polygon use a segment shaped like their bounding box."
    )
x["hash"] = self.get_cache_hash()
save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
return x
```

**[6] `PolygonSemanticDataset.load_mask`** (ultralytics/data/dataset.py, lines 1137–1158)

```python
def load_mask(self, index: int, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Rasterize this image's polygons into a (H, W) uint8 semantic mask, bg = self.bg_class_idx."""
    h, w = image_shape
    label = self.labels[index]
    cls = label.get("cls")
    segments = label.get("segments") or []
    if cls is None or len(cls) == 0 or len(segments) == 0:
        return np.full((h, w), self.bg_class_idx, dtype=np.uint8)

    # Denormalize polygons (stored as normalized xy) to pixel coordinates at (h, w).
    scale = np.array([w, h], dtype=np.float32)
    polys = [np.asarray(s, dtype=np.float32).reshape(-1, 2) * scale for s in segments]
    # Returns (H, W) instance index map: 0 = no polygon, 1..N = sorted instance index.
    inst, sorted_idx = polygons2masks_overlap((h, w), polys, downsample_ratio=1)
    out = np.full((h, w), self.bg_class_idx, dtype=np.uint8)
    fg = inst > 0
    if int(self.data.get("nc", 0)) == 1:  # binary: fg=1 regardless of label cls value
        out[fg] = 1
    else:
        cls_arr = np.asarray(cls).reshape(-1).astype(np.int32)[sorted_idx]
        out[fg] = cls_arr[inst[fg] - 1].astype(np.uint8)
    return out
```

</details>

## 5. `predict-returns-list`

**Question:** I get AttributeError: 'list' object has no attribute 'boxes' after calling the model. What does prediction return?

**Key facts:**

- [x] Calling the model (an alias of Model.predict) returns a list of Results objects, one per image or frame.
- [x] Index or iterate the list to access detections, e.g. results[0].boxes.

<details><summary>Sources</summary>

**[1] `ultralytics.engine.predictor`** (ultralytics/engine/predictor.py, lines 1–607)

```python
STREAM_WARNING = """
Inference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large
sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""
def _prefetch(iterator):
    """Yield items while loading the next one on a worker thread."""
    ...
class BasePredictor():
    """A base class for creating predictors."""
```

**[2] `FastSAM.predict`** (ultralytics/models/fastsam/model.py, lines 49–78)

```python
def predict(
    self,
    source=None,
    stream: bool = False,
    bboxes: list | None = None,
    points: list | None = None,
    labels: list | None = None,
    texts: str | list[str] | None = None,
    **kwargs: Any,
):
    """Perform segmentation prediction on image or video source.

    Supports prompted segmentation with bounding boxes, points, labels, and texts. The method packages these prompts
    and passes them to the parent class predict method for processing.

    Args:
        source (str | PIL.Image | np.ndarray, optional): Input source for prediction, can be a file path, URL, PIL
            image, or numpy array. Defaults to the package sample assets when None.
        stream (bool): Whether to enable real-time streaming mode for video inputs.
        bboxes (list, optional): Bounding box coordinates for prompted segmentation in format [[x1, y1, x2, y2]].
        points (list, optional): Point coordinates for prompted segmentation in format [[x, y]].
        labels (list, optional): Point prompt labels, 1 for foreground and 0 for background; all 1 if omitted.
        texts (str | list[str], optional): Text prompts for segmentation guidance.
        **kwargs (Any): Additional keyword arguments passed to the predictor.

    Returns:
        (list): List of Results objects containing the prediction results.
    """
    prompts = {"bboxes": bboxes, "points": points, "labels": labels, "texts": texts}
    return super().predict(source, stream, prompts=prompts, **kwargs)
```

**[3] `Boxes.cls`** (ultralytics/engine/results.py, lines 1023–1037)

```python
@property
def cls(self) -> torch.Tensor | np.ndarray:
    """Return the class ID tensor representing category predictions for each bounding box.

    Returns:
        (torch.Tensor | np.ndarray): A tensor or numpy array containing the class IDs for each detection box. The
            shape is (N,), where N is the number of boxes.

    Examples:
        >>> results = model("image.jpg")
        >>> boxes = results[0].boxes
        >>> class_ids = boxes.cls
        >>> print(class_ids)  # tensor([0., 2., 1.])
    """
    return self.data[:, -1]
```

**[4] `Model.__call__`** (ultralytics/engine/model.py, lines 143–171)

```python
def __call__(
    self,
    source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> Iterator[Results | torch.Tensor] | list[Results] | list[torch.Tensor]:
    """Alias for the predict method, enabling the model instance to be callable for predictions.

    This method simplifies the process of making predictions by allowing the model instance to be called directly
    with the required arguments.

    Args:
        source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple): The source of the image(s)
            to make predictions on. Can be a file path, URL, PIL image, numpy array, PyTorch tensor, or a list/tuple
            of these.
        stream (bool): If True, treat the input source as a continuous stream for predictions.
        **kwargs (Any): Additional keyword arguments to configure the prediction process.

    Returns:
        (Iterator[ultralytics.engine.results.Results | torch.Tensor] | list[ultralytics.engine.results.Results] |
        list[torch.Tensor]): Prediction results or embeddings, streamed when `stream=True`.

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> results = model("https://ultralytics.com/images/bus.jpg")
        >>> for r in results:
        ...     print(f"Detected {len(r)} objects in image")
    """
    return self.predict(source, stream, **kwargs)
```

**[5] `SAM.predict`** (ultralytics/models/sam/model.py, lines 86–110)

```python
def predict(self, source, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
    """Perform segmentation prediction on the given image or video source.

    Args:
        source (str | PIL.Image | np.ndarray): Path to the image or video file, or a PIL.Image object, or a
            np.ndarray object.
        stream (bool): If True, enables real-time streaming.
        bboxes (list[list[float]] | None): List of bounding box coordinates for prompted segmentation.
        points (list[list[float]] | None): List of points for prompted segmentation.
        labels (list[int] | None): List of labels for prompted segmentation.
        **kwargs (Any): Additional keyword arguments for prediction.

    Returns:
        (list[Results] | Generator[Results]): The model predictions, as a generator if `stream=True`.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        ...     print(f"Detected {len(r.masks)} masks")
    """
    overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "imgsz": 1024}
    kwargs = {**overrides, **kwargs, "retina_masks": True}
    prompts = {"bboxes": bboxes, "points": points, "labels": labels}
    return super().predict(source, stream, prompts=prompts, **kwargs)
```

**[6] `SAM.__call__`** (ultralytics/models/sam/model.py, lines 112–135)

```python
def __call__(self, source=None, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
    """Perform segmentation prediction on the given image or video source.

    This method is an alias for the 'predict' method, providing a convenient way to call the SAM model for
    segmentation tasks.

    Args:
        source (str | PIL.Image | np.ndarray | None): Path to the image or video file, or a PIL.Image object, or a
            np.ndarray object.
        stream (bool): If True, enables real-time streaming.
        bboxes (list[list[float]] | None): List of bounding box coordinates for prompted segmentation.
        points (list[list[float]] | None): List of points for prompted segmentation.
        labels (list[int] | None): List of labels for prompted segmentation.
        **kwargs (Any): Additional keyword arguments to be passed to the predict method.

    Returns:
        (list[Results] | Generator[Results]): The model predictions, as a generator if `stream=True`.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam("image.jpg", points=[[500, 375]])
        >>> print(f"Detected {len(results[0].masks)} masks")
    """
    return self.predict(source, stream, bboxes, points, labels, **kwargs)
```

</details>

## 6. `predict-box-size`

**Question:** How do I get the width and height in pixels of each detected object?

**Key facts:**

- [x] Use results[0].boxes.xywh, which gives [x_center, y_center, width, height] for each box.
- [x] The values are in pixels of the original image, so width = xywh[:, 2] and height = xywh[:, 3] (xywhn is the normalized version).

<details><summary>Sources</summary>

**[1] `Boxes`** (ultralytics/engine/results.py, lines 935–1123)

```python
class Boxes(BaseTensor):
    """A class for managing and manipulating detection boxes.

    This class provides comprehensive functionality for handling detection boxes, including their coordinates,
    confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for
    easy manipulation and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | np.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | np.ndarray): Confidence scores for each box.
        cls (torch.Tensor | np.ndarray): Class labels for each box.
        id (torch.Tensor | None): Tracking IDs for each box (if available).
        xywh (torch.Tensor | np.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | np.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | np.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu: Return a copy of the object with all tensors on CPU memory.
        numpy: Return a copy of the object with all tensors as numpy arrays.
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """
    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize the Boxes class with detection box data and the original image shape."""
        ...
    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """Return bounding boxes in [x1, y1, x2, y2] format."""
        ...
    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """Return the confidence scores for each detection box."""
        ...
    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """Return the class ID tensor representing category predictions for each bounding box."""
        ...
    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        """Return the tracking IDs for each detection box if available."""
        ...
    @cached_property
    def xywh(self) -> torch.Tensor | np.ndarray:
        """Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format."""
        ...
    @cached_property
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        """Return normalized bounding box coordinates relative to the original image size."""
        ...
    @cached_property
    def xywhn(self) -> torch.Tensor | np.ndarray:
        """Return normalized bounding boxes in [x, y, width, height] format."""
        ...
```

**[2] `visualize_image_annotations`** (ultralytics/data/utils.py, lines 516–557)

```python
def visualize_image_annotations(image_path: str, txt_path: str, label_map: dict[int, str]):
    """Visualize YOLO detection annotations (bounding boxes and class labels) on an image.

    This function reads an image and its corresponding YOLO detection label file, then draws bounding boxes around
    detected objects and labels them with their respective class names. The bounding box colors are assigned based on
    the class ID, and the text color is dynamically adjusted for readability, depending on the background color's
    luminance.

    Args:
        image_path (str): Path to the image file to annotate. The file must be readable by PIL.
        txt_path (str): Path to a YOLO detection label file with one `class x_center y_center width height` line per
            object. Segmentation polygon and pose label rows are not supported.
        label_map (dict[int, str]): A dictionary that maps class IDs (integers) to class labels (strings).

    Examples:
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # Should include all annotated classes
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)
    """
    import matplotlib.pyplot as plt

    from ultralytics.utils.plotting import colors

    img = np.array(ImageOps.exif_transpose(Image.open(image_path)))  # upright, as dataloaders read it for training
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    _, ax = plt.subplots(1)  # Plot the image and annotations
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, False))  # Get and normalize an RGB color for Matplotlib
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")  # Create a rectangle
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
        ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    plt.show()
```

**[3] `Boxes.xywhn`** (ultralytics/engine/results.py, lines 1102–1123)

```python
@cached_property
def xywhn(self) -> torch.Tensor | np.ndarray:
    """Return normalized bounding boxes in [x, y, width, height] format.

    This property calculates and returns the normalized bounding box coordinates in the format [x_center, y_center,
    width, height], where all values are relative to the original image dimensions.

    Returns:
        (torch.Tensor | np.ndarray): Normalized bounding boxes with shape (N, 4), where N is the number of boxes.
            Each row contains [x_center, y_center, width, height] values normalized to [0, 1] based on the original
            image dimensions.

    Examples:
        >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
        >>> normalized = boxes.xywhn
        >>> print(normalized)
        tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
    """
    xywh = ops.xyxy2xywh(self.xyxy)
    xywh[..., [0, 2]] /= self.orig_shape[1]
    xywh[..., [1, 3]] /= self.orig_shape[0]
    return xywh
```

**[4] `Results`** (ultralytics/engine/results.py, lines 197–932)

```python
class Results(SimpleClass, DataExportMixin):
    """A class for storing and manipulating inference results.

    This class provides comprehensive functionality for handling inference results from various Ultralytics models,
    including detection, instance segmentation, semantic segmentation, depth estimation, classification, pose
    estimation, and oriented bounding box detection. It supports visualization, data export, and various coordinate
    transformations.

    Attributes:
        orig_img (np.ndarray): The original image as a numpy array.
        orig_shape (tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Detected bounding boxes.
        masks (Masks | None): Segmentation masks.
        probs (Probs | None): Classification probabilities.
        keypoints (Keypoints | None): Detected keypoints.
        obb (OBB | None): Oriented bounding boxes.
        semantic_mask (SemanticMask | None): Semantic segmentation class map.
        depth (DepthMap | None): Per-pixel depth map.
        speed (dict): Dictionary containing inference speed information.
        names (dict): Dictionary mapping class indices to class names.
        path (str): Path to the input image file.
        save_dir (str | None): Directory to save results.

    Methods:
        update: Update the Results object with new detection data.
        cpu: Return a copy of the Results object with all tensors moved to CPU memory.
        numpy: Convert all tensors in the Results object to numpy arrays.
        cuda: Move all tensors in the Results object to GPU memory.
        to: Move all tensors to the specified device and dtype.
        new: Create a new Results object with the same image, path, names, and speed attributes.
        plot: Plot detection results on an input BGR image.
        show: Display the image with annotated inference results.
        save: Save annotated inference results image to file.
        verbose: Return a log string for each task in the results.
        save_txt: Save detection results to a text file.
        save_crop: Save cropped detection images to specified directory.
        summary: Convert inference results to a summarized dictionary.
        to_df: Convert detection results to a Polars DataFrame.
        to_json: Convert detection results to JSON format.
        to_csv: Convert detection results to a CSV format.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # Get the first result
        >>> boxes = result.boxes  # Get the boxes for the first result
        >>> masks = result.masks  # Get the masks for the first result
        >>> for result in results:
        ...     result.plot()  # Plot detection results
    """
    def __init__(self, orig_img: np.ndarray, path: str, names: dict[int, str], boxes: torch.Tensor | None=None, masks: torch.Tensor | None=None, probs: torch.Tensor | None=None, keypoints: torch.Tensor | None=None, obb: torch.Tensor | None=None, speed: dict[str, float] | None=None, semantic_mask: torch.Tensor | None=None, depth: torch.Tensor | None=None) -> None:
        """Initialize the Results class for storing and manipulating inference results."""
        ...
    def __getitem__(self, idx):
        """Return a Results object for a specific index of inference results."""
        ...
    def __len__(self) -> int:
        """Return the number of results in the Results object."""
        ...
    def update(self, boxes: torch.Tensor | None=None, masks: torch.Tensor | None=None, probs: torch.Tensor | None=None, obb: torch.Tensor | None=None, keypoints: torch.Tensor | None=None, semantic_mask: torch.Tensor | None=None, depth: torch.Tensor | None=None):
        """Update the Results object with new detection data."""
        ...
    def _apply(self, fn: str, *args, **kwargs):
        """Apply a function to all non-empty attributes and return a new Results object with modified attributes."""
        ...
```

**[5] `BaseDataset`** (ultralytics/data/base.py, lines 25–503)

```python
class BaseDataset(Dataset):
    """Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.

    Attributes:
        img_path (str | list[str]): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float | int): Dataset ratio or image count to use.
        channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with OpenCV
            are in BGR channel order.
        cv2_flag (int): OpenCV flag for reading images.
        im_files (list[str]): List of image file paths.
        labels (list[dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (list[Path]): List of numpy file paths.
        cache (str | None): Cache setting ('ram', 'disk', or None for no caching).
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Sort images by aspect ratio and set batch shapes for rectangular training.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """
```

**[6] `get_window_obj`** (ultralytics/data/split_dota.py, lines 155–176)

```python
def get_window_obj(anno: dict[str, Any], windows: np.ndarray, iof_thr: float = 0.7) -> list[np.ndarray]:
    """Get objects for each window based on IoF threshold.

    Args:
        anno (dict[str, Any]): Annotation dict with 'ori_size' (H, W) and 'label' (N, 9) keys, where each label row is
            [cls, x1, y1, x2, y2, x3, y3, x4, y4] in normalized coordinates. The labels are denormalized in place.
        windows (np.ndarray): Array of window coordinates with shape (M, 4) as [x_start, y_start, x_stop, y_stop].
        iof_thr (float, optional): IoF threshold at or above which an object is assigned to a window.

    Returns:
        (list[np.ndarray]): List of M label arrays, one per window, with pixel coordinates in the original image.
    """
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        label[:, 1::2] *= w
        label[:, 2::2] *= h
        iofs = bbox_iof(label[:, 1:], windows)
        # Unnormalized and misaligned coordinates
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # window_anns
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # window_anns
```

</details>

## 7. `predict-plot-no-boxes`

**Question:** How can I draw only the segmentation masks on the output image, without the bounding boxes?

**Key facts:**

- [x] Call result.plot(boxes=False): masks are still drawn (masks=True by default) but bounding boxes are not.
- [ ] When saving or showing predictions, predict(..., show_boxes=False) has the same effect.

<details><summary>Sources</summary>

**[1] `SegmentationPredictor.postprocess`** (ultralytics/models/yolo/segment/predict.py, lines 48–66)

```python
def postprocess(self, preds, img, orig_imgs):
    """Apply non-max suppression and process segmentation detections for each image in the input batch.

    Args:
        preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
        img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
        orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

    Returns:
        (list): List of Results objects containing the segmentation predictions for each image in the batch. Each
            Results object includes both bounding boxes and segmentation masks.

    Examples:
        >>> predictor = SegmentationPredictor(overrides=dict(model="yolo26n-seg.pt"))
        >>> results = predictor.postprocess(preds, img, orig_img)
    """
    # Extract protos - tuple if PyTorch model or array if exported
    protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
    return super().postprocess(preds[0], img, orig_imgs, protos=protos)
```

**[2] `Results.plot`** (ultralytics/engine/results.py, lines 487–571)

```python
def plot(
    self,
    conf: bool = True,
    line_width: float | None = None,
    font_size: float | None = None,
    font: str = "Arial.ttf",
    pil: bool = False,
    img: np.ndarray | torch.Tensor | None = None,
    kpt_radius: int = 5,
    kpt_line: bool = True,
    labels: bool = True,
    boxes: bool = True,
    masks: bool = True,
    probs: bool = True,
    show: bool = False,
    save: bool = False,
    filename: str | None = None,
    color_mode: str = "class",
    txt_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Plot detection results on an input BGR image.

    Args:
        conf (bool): Whether to plot detection confidence scores.
        line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
        font_size (float | None): Font size for text. If None, scaled to image size.
        font (str): Font to use for text.
        pil (bool): Whether to return the image as a PIL Image.
        img (np.ndarray | torch.Tensor | None): Image to plot on. Tensor images must be contiguous HWC BGR uint8. If
            None, uses the original image.
        kpt_radius (int): Radius of drawn keypoints.
        kpt_line (bool): Whether to draw lines connecting keypoints.
        labels (bool): Whether to plot labels of bounding boxes.
        boxes (bool): Whether to plot bounding boxes.
        masks (bool): Whether to plot masks.
        probs (bool): Whether to plot classification probabilities.
        show (bool): Whether to display the annotated image.
        save (bool): Whether to save the annotated image.
        filename (str | None): Filename to save image if save is True.
        color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.
        txt_color (tuple[int, int, int]): Text color in BGR format for classification output.

    Returns:
        (np.ndarray | PIL.Image.Image): Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`.

    Examples:
        >>> results = model("image.jpg")
        >>> for result in results:
        ...     im = result.plot(pil=True)
        ...     im.show()
    """
    assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
    if img is None and isinstance(self.orig_img, torch.Tensor):
        img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

    names = self.names
    is_obb = self.obb is not None
    pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
    pred_masks, show_masks = self.masks, masks
    pred_probs, show_probs = self.probs, probs
    if pred_boxes is not None and (show_boxes or (pred_masks and show_masks)):
        pred_boxes = pred_boxes.cpu()  # one host transfer avoids per-box GPU syncs in the color and label loops
    annotator = Annotator(
        deepcopy(self.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )

    # Plot Segment results
    if pred_masks and show_masks:
        pred_mask_data = torch.as_tensor(pred_masks.data)  # no-op for torch, converts a numpy() result
        idx = (
            pred_boxes.id
            if pred_boxes and pred_boxes.is_track and color_mode == "instance"
            else pred_boxes.cls
            if pred_boxes and color_mode == "class"
            else reversed(range(len(pred_masks)))
        )
        annotator.masks(pred_mask_data, colors=[colors(x, True) for x in idx])

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
```

**[3] `SegmentationPredictor.construct_result`** (ultralytics/models/yolo/segment/predict.py, lines 86–111)

```python
def construct_result(self, pred, img, orig_img, img_path, proto):
    """Construct a single result object from the prediction.

    Args:
        pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
        img (torch.Tensor): The image after preprocessing.
        orig_img (np.ndarray): The original image before preprocessing.
        img_path (str): The path to the original image.
        proto (torch.Tensor): The prototype masks.

    Returns:
        (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
    """
    if pred.shape[0] == 0:  # save empty boxes
        masks = None
    elif self.args.retina_masks:
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
    else:
        masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    if masks is not None:
        keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
        if not (all(keep) or getattr(self, "_feats", None) is not None):  # skip filter if native ReID enabled
            pred, masks = pred[keep], masks[keep]  # indexing is slow
    return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
```

**[4] `Predictor.inference`** (ultralytics/models/sam/predict.py, lines 181–218)

```python
def inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
    """Perform image segmentation inference based on the given input cues, using the currently loaded image.

    This method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt encoder,
    and mask decoder for real-time and promptable segmentation tasks.

    Args:
        im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
        bboxes (np.ndarray | list | None): Bounding boxes with shape (N, 4), in XYXY format.
        points (np.ndarray | list | None): Points indicating object locations with shape (N, 2), in pixels.
        labels (np.ndarray | list | None): Labels for point prompts, shape (N,). 1 = foreground, 0 = background.
        masks (np.ndarray | None): Low-resolution masks from previous predictions, shape (N, H, W). For SAM H=W=256.
        multimask_output (bool): Flag to return multiple masks. Helpful for ambiguous prompts.
        *args (Any): Additional positional arguments.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        pred_masks (torch.Tensor): The output masks in shape (C, H, W), where C is the number of generated masks.
        pred_scores (torch.Tensor): An array of length C containing quality scores predicted by the model for each
            mask.
        pred_bboxes (torch.Tensor): Bounding boxes with shape (C, 4), only returned when no prompts are given and
            the whole image is segmented via `generate`.

    Examples:
        >>> predictor = Predictor(overrides=dict(model="sam_b.pt"))
        >>> predictor.set_image("image.jpg")
        >>> results = predictor(bboxes=[[0, 0, 100, 100]])
    """
    # Override prompts if any stored in self.prompts
    bboxes = self.prompts.pop("bboxes", bboxes)
    points = self.prompts.pop("points", points)
    masks = self.prompts.pop("masks", masks)
    labels = self.prompts.pop("labels", labels)

    if all(i is None for i in [bboxes, points, masks]):
        return self.generate(im, *args, **kwargs)

    return self.prompt_inference(im, bboxes, points, labels, masks, multimask_output)
```

**[5] `SegmentationPredictor.__init__`** (ultralytics/models/yolo/segment/predict.py, lines 34–46)

```python
def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
    """Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Args:
        cfg (dict): Configuration for the predictor.
        overrides (dict, optional): Configuration overrides that take precedence over cfg.
        _callbacks (dict, optional): Dictionary of callback functions to be invoked during prediction.
    """
    super().__init__(cfg, overrides, _callbacks)
    self.args.task = "segment"
```

**[6] `Predictor.postprocess`** (ultralytics/models/sam/predict.py, lines 512–569)

```python
def postprocess(self, preds, img, orig_imgs):
    """Post-process SAM's inference outputs to generate object detection masks and bounding boxes.

    This method scales masks and boxes to the original image size and applies a threshold to the mask
    predictions. It leverages SAM's advanced architecture for real-time, promptable segmentation tasks.

    Args:
        preds (tuple): The output from SAM model inference, containing:
            - pred_masks (torch.Tensor): Predicted masks with shape (N, H, W).
            - pred_scores (torch.Tensor): Confidence scores for each mask with shape (N,).
            - pred_bboxes (torch.Tensor, optional): Predicted bounding boxes if segment_all is True.
        img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
        orig_imgs (list[np.ndarray] | torch.Tensor): The original, unprocessed images.

    Returns:
        (list[Results]): List of Results objects containing detection masks, bounding boxes, and other metadata for
            each processed image.

    Examples:
        >>> predictor = Predictor()
        >>> preds = predictor.inference(img)
        >>> results = predictor.postprocess(preds, img, orig_imgs)
    """
    # (N, H, W), (N,)
    pred_masks, pred_scores = preds[:2]
    pred_bboxes = preds[2] if self.segment_all else None
    names = dict(enumerate(str(i) for i in range(pred_masks.shape[0])))

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

    results = []
    for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
        if masks.shape[0] == 0:
            masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
        else:
            idx = pred_scores > self.args.conf
            gain = min(i / o for i, o in zip(img.shape[2:], orig_img.shape[:2]))  # letterbox gain at the input
            ratio = [
                round(o * gain) * m / i / o for o, m, i in zip(orig_img.shape[:2], masks.shape[1:], img.shape[2:])
            ]
            masks = ops.scale_masks(masks[idx][None].float(), orig_img.shape[:2], (ratio, (0, 0)), padding=False)[0]
            if self.non_overlap_masks:
                masks = self.model._apply_non_overlapping_constraints(masks[:, None])[:, 0]
            masks = masks > self.model.mask_threshold  # to bool
            if pred_bboxes is not None:
                pred_bboxes = ops.scale_boxes(
                    img.shape[2:], pred_bboxes[idx].float(), orig_img.shape, padding=False
                )
            else:
                pred_bboxes = batched_mask_to_box(masks)
            # NOTE: SAM models do not return cls info. This `cls` here is just a placeholder for consistency.
            cls = torch.arange(pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device)[idx]
            pred_bboxes = torch.cat([pred_bboxes, pred_scores[idx, None], cls[:, None]], dim=-1)
        results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
    # Reset segment-all mode.
    self.segment_all = False
    return results
```

</details>

## 8. `val-vs-predict`

**Question:** Why do model.val() and model.predict() give different detections on the same images?

**Key facts:**

- [ ] predict() uses a default confidence threshold of conf=0.25.
- [ ] val() uses conf=0.001 by default (0.01 for OBB) to keep low-confidence boxes for computing mAP.
- [ ] Passing the same conf (and other arguments) explicitly to both makes their outputs comparable.

<details><summary>Sources</summary>

**[1] `Model`** (ultralytics/engine/model.py, lines 35–1275)

```python
class Model(torch.nn.Module):
    """A base class for implementing YOLO models, unifying APIs across different model types.

    This class provides a common interface for various operations related to YOLO models, such as training, validation,
    prediction, exporting, and benchmarking. It handles different types of models, including those loaded from local
    files or Triton Server.

    Attributes:
        callbacks (dict): A dictionary of callback functions for various events during model operations.
        predictor (BasePredictor): The predictor object used for making predictions.
        model (torch.nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        ckpt (dict): The checkpoint data of the loaded weights, if any.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (dict): A dictionary of overrides for model configuration.
        metrics (ultralytics.utils.metrics.DetMetrics): The latest training/validation metrics.
        task (str): The type of task the model is intended for.
        model_name (str): The name of the model.

    Methods:
        __call__: Alias for the predict method, enabling the model instance to be callable.
        _new: Initialize a new model based on a configuration file.
        _load: Load a model from a checkpoint file.
        _check_is_pytorch_model: Ensure that the model is a PyTorch model.
        reset_weights: Reset the model's weights to their initial state.
        load: Load model weights from a specified file.
        save: Save the current state of the model to a file.
        info: Log or return information about the model.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimized inference.
        embed: Generate image embeddings from intermediate model layers.
        predict: Perform predictions on given image sources.
        track: Perform object tracking.
        val: Validate the model on a dataset.
        calibrate: Fit scale-only depth calibration for depth models.
        benchmark: Benchmark the model on various export formats.
        export: Export the model to different formats.
        train: Train the model on a dataset.
        tune: Perform hyperparameter tuning.
        _apply: Apply a function to the model's tensors.
        add_callback: Add a callback function for an event.
        clear_callback: Clear all callbacks for an event.
        reset_callbacks: Reset all callbacks to their default functions.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.predict("image.jpg")
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> metrics = model.val()
        >>> model.export(format="onnx")
    """
    def __init__(self, model: str | Path | Model='yolo26n.pt', task: str | None=None, verbose: bool=False) -> None:
        """Initialize a new instance of the YOLO model class."""
        ...
    def __call__(self, source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor | None=None, stream: bool=False, **kwargs: Any) -> Iterator[Results | torch.Tensor] | list[Results] | list[torch.Tensor]:
        """Alias for the predict method, enabling the model instance to be callable for predictions."""
        ...
    @staticmethod
    def is_triton_model(model: str) -> bool:
        """Check if the given model string is a Triton Server URL."""
        ...
    def _new(self, cfg: str, task: str | None=None, verbose: bool=False) -> None:
        """Initialize a new model and infer the task type from model definitions."""
        ...
    def _load(self, weights: str, task: str | None=None) -> None:
        """Load a model from a checkpoint file or initialize it from a weights file."""
        ...
```

**[2] `Model.val`** (ultralytics/engine/model.py, lines 613–647)

```python
def val(
    self,
    validator=None,
    **kwargs: Any,
):
    """Validate the model using a specified dataset and validation configuration.

    This method facilitates the model validation process, allowing for customization through various settings. It
    supports validation with a custom validator or the default validation approach. The method combines default
    configurations, method-specific defaults, and user-provided arguments to configure the validation process.

    Args:
        validator (type[ultralytics.engine.validator.BaseValidator], optional): A custom validator class for
            validating the model. If None, uses the default validator for the task.
        **kwargs (Any): Arbitrary keyword arguments for customizing the validation process.

    Returns:
        (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process. The
            specific metrics type depends on the task (e.g., DetMetrics, SegmentMetrics,
            ClassifyMetrics, PoseMetrics).

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.val(data="coco8.yaml", imgsz=640)
        >>> print(results.box.map)  # Print mAP50-95
    """
    custom = {"rect": True}  # method defaults
    if kwargs.get("data") is None:
        kwargs["data"] = self._val_data()
    args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

    validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
    validator(model=self.model)
    self.metrics = validator.metrics
    return validator.metrics
```

**[3] `YOLOE`** (ultralytics/models/yolo/model.py, lines 214–567)

```python
class YOLOE(Model):
    """YOLOE object detection and segmentation model.

    YOLOE is an enhanced YOLO model that supports both object detection and instance segmentation tasks with improved
    performance and additional features like visual and text prompt embeddings.

    Attributes:
        model: The loaded YOLOE model instance.
        task: The task type (detect or segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOE model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        get_text_pe: Get text prompt embeddings for the given texts.
        get_visual_pe: Get visual prompt embeddings for the given image and visual features.
        set_vocab: Set vocabulary and class names for the YOLOE model.
        get_vocab: Get the vocabulary for the given class names, which become the model's classes as the head is fused.
        set_classes: Set the model's class names and embeddings for detection.
        save_prompt_embeddings: Save the current prompt embeddings and class names to an NPZ file.
        load_prompt_embeddings: Load prompt embeddings and class names from an NPZ file.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a YOLOE segmentation model
        >>> model = YOLOE("yoloe-11s-seg.pt")

        Predict with visual prompts, whose 'cls' holds one class index per box
        >>> from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        >>> prompts = {"bboxes": np.array([[10, 20, 100, 200]]), "cls": np.array([0])}
        >>> results = model.predict("image.jpg", visual_prompts=prompts, predictor=YOLOEVPSegPredictor)

        Re-parameterize into a prompt-free model, which no longer accepts prompts
        >>> names = ["person", "car", "dog"]
        >>> model.set_vocab(model.get_vocab(names), names)
    """
    def __init__(self, model: str | Path='yoloe-11s-seg.pt', task: str | None=None, verbose: bool=False) -> None:
        """Initialize YOLOE model with a pre-trained model file."""
        ...
    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        ...
    def get_text_pe(self, texts):
        """Get text prompt embeddings for the given texts."""
        ...
    def get_visual_pe(self, img, visual):
        """Get visual prompt embeddings for the given image and visual features."""
        ...
    def set_vocab(self, vocab: torch.nn.ModuleList, names: list[str], one2one_vocab: torch.nn.ModuleList | None=None) -> None:
        """Re-parameterize the model into a prompt-free one over the given class names."""
        ...
    def get_vocab(self, names):
        """Get the vocabulary for the given class names, which become the model's classes as the head is fused."""
        ...
    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None=None) -> None:
        """Set the model's class names and embeddings for detection."""
        ...
    def _prompt_embedding_model(self) -> str:
        """Return the checkpoint identifier used to bind prompt embeddings to this model."""
        ...
    def save_prompt_embeddings(self, file: str | Path) -> Path:
        """Save the current prompt embeddings and class names to an NPZ file."""
        ...
    def load_prompt_embeddings(self, file: str | Path) -> None:
        """Load prompt embeddings and class names from a model-bound NPZ file."""
        ...
    def val(self, validator=None, load_vp: bool=False, refer_data: str | None=None, **kwargs):
        """Validate the model using text or visual prompts."""
        ...
```

**[4] `NASValidator`** (ultralytics/models/nas/val.py, lines 11–35)

```python
class NASValidator(DetectionValidator):
    """Ultralytics YOLO NAS Validator for object detection.

    Extends DetectionValidator from the Ultralytics models package and is designed to post-process the raw predictions
    generated by YOLO NAS models. It performs non-maximum suppression to remove overlapping and low-confidence boxes,
    ultimately producing the final detections.

    Attributes:
        args (SimpleNamespace): Namespace containing various configurations for post-processing, such as confidence and
            IoU thresholds.

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> metrics = model.val(data="coco8.yaml")  # runs NASValidator internally

    Notes:
        This class is generally not instantiated directly but is used internally within the NAS class.
    """
    def postprocess(self, preds_in):
        """Apply Non-maximum suppression to prediction outputs."""
        ...
```

**[5] `WorldValidator`** (ultralytics/models/yolo/world/val.py, lines 12–40)

```python
class WorldValidator(DetectionValidator):
    """A validator for YOLO-World models that sets dataset class names before validation.

    Open-vocabulary YOLO-World models default to 80 COCO classes, so validating on a dataset with different classes
    (e.g. LVIS) fails or yields zero metrics. This validator generates text embeddings for the dataset's class names so
    standalone `model.val()` works. During training, classes are set via the `on_pretrain_routine_end` callback instead.
    """
    def __call__(self, trainer=None, model=None):
        """Set dataset classes for standalone validation, then run validation."""
        ...
```

**[6] `NAS`** (ultralytics/models/nas/model.py, lines 20–94)

```python
class NAS(Model):
    """YOLO-NAS model for object detection.

    This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine. It
    is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

    Attributes:
        model (torch.nn.Module): The loaded YOLO-NAS model.
        task (str): The task type for the model, defaults to 'detect'.
        predictor (NASPredictor): The predictor instance for making predictions, created on the first prediction.

    Methods:
        info: Log model information and return model details.

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

    Notes:
        YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.
    """
    def __init__(self, model: str='yolo_nas_s.pt') -> None:
        """Initialize the NAS model with the provided or default model."""
        ...
    def _load(self, weights: str, task=None) -> None:
        """Load an existing NAS model weights or create a new NAS model with pretrained weights."""
        ...
    def info(self, detailed: bool=False, verbose: bool=True, imgsz: int | list[int]=640) -> tuple:
        """Log model information."""
        ...
    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Return a dictionary mapping tasks to respective predictor and validator classes."""
        ...
```

</details>

## 9. `val-test-split`

**Question:** How do I evaluate my trained model on the test split instead of the validation split?

**Key facts:**

- [x] Call model.val(data='data.yaml', split='test').
- [ ] The validator loads the dataset with check_det_dataset and builds the dataloader from the entry named by split.
- [ ] The dataset YAML must define a test entry; otherwise the images are not found.

<details><summary>Sources</summary>

**[1] `ultralytics.engine.validator`** (ultralytics/engine/validator.py, lines 1–435)

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo val model=yolo26n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo val model=yolo26n.pt                 # PyTorch
                     yolo26n.torchscript        # TorchScript
                     yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                     yolo26n_openvino_model     # OpenVINO
                     yolo26n.engine             # TensorRT
                     yolo26n.mlpackage          # CoreML (macOS-only)
                     yolo26n.aimodel            # Apple Core AI
                     yolo26n_saved_model        # TensorFlow SavedModel
                     yolo26n.pb                 # TensorFlow GraphDef
                     yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                     yolo26n.tflite             # LiteRT
                     yolo26n_paddle_model       # PaddlePaddle
                     yolo26n.mnn                # MNN
                     yolo26n_ncnn_model         # NCNN
                     yolo26n_imx_model          # Sony IMX
                     yolo26n_rknn_model         # Rockchip RKNN
                     yolo26n_executorch_model   # ExecuTorch
                     yolo26n_axelera_model      # Axelera AI
                     yolo26n_deepx_model        # DEEPX
                     yolo26n_qnn.onnx           # Qualcomm QNN
                     yolo26n_hailo_model        # Hailo
                     yolo26n_ascend_model       # Huawei Ascend
                     yolo26n_xilinx_model       # AMD Xilinx
"""
from __future__ import annotations
import json
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOCAL_RANK, LOGGER, RANK, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile, linear_sum_assignment
from ultralytics.utils.torch_utils import (
    attempt_compile,
    autocast,
    get_torch_device_backend,
    select_device,
    smart_inference_mode,
    torch_distributed_zero_first,
    unwrap_model,
)
class BaseValidator():
    """A base class for creating validators."""
```

**[2] `WorldTrainerFromScratch.final_eval`** (ultralytics/models/yolo/world/train_world.py, lines 186–194)

```python
def final_eval(self):
    """Perform final evaluation and validation for the YOLO-World model.

    Configures the validator with appropriate dataset and split information before running evaluation.
    """
    val = self.args.data["val"]["yolo_data"][0]
    self.validator.args.data = val
    self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
    return super().final_eval()
```

**[3] `Model.eval`** (ultralytics/engine/model.py, lines 1236–1252)

```python
def eval(self):
    """Set the model to evaluation mode.

    This method changes the model's mode to evaluation, which affects layers like dropout and batch normalization
    that behave differently during training and evaluation. In evaluation mode, these layers use running statistics
    rather than computing batch statistics, and dropout layers are disabled.

    Returns:
        (Model): The model instance with evaluation mode set.

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> model.eval()
        >>> # Model is now in evaluation mode for inference
    """
    self.model.eval()
    return self
```

**[4] `DepthTrainer.final_eval`** (ultralytics/models/yolo/depth/train.py, lines 132–175)

```python
def final_eval(self) -> None:
    """Run the standard final evaluation, then calibrate the saved checkpoints.

    After training, fits the scale-only log-affine (``cal_a``/``cal_b``) on the validation
    set and writes it into best.pt/last.pt, so the model outputs metric-scaled depth out of
    the box. When ``plots`` is set, also writes ``val_batch{ni}_calibrated.jpg``
    (RGB | GT | raw | calibrated) comparison panels.
    """
    super().final_eval()
    if RANK not in {-1, 0}:
        return
    try:
        from .calibrate import calibrate_checkpoint

        LOGGER.info("Calibrating depth output scale on the validation set...")
        plot_ckpt = self.best if self.best.exists() else self.last
        for ckpt in (self.best, self.last):
            if ckpt.exists():
                plot_dir = self.save_dir if self.args.plots and ckpt == plot_ckpt else None
                validation_path = self.data[self.args.split]
                validation_split = None
                if isinstance(validation_path, (str, Path)):
                    try:
                        validation_split = (
                            Path(validation_path)
                            .resolve()
                            .relative_to(Path(self.data["path"]).resolve())
                            .as_posix()
                        )
                    except ValueError:
                        pass  # External validation paths have no portable dataset-root-relative identifier.
                provenance = calibrate_checkpoint(
                    ckpt,
                    self.test_loader,
                    self.device,
                    plot_dir=plot_dir,
                    dataset_hash=self.data.get("hash"),
                    validation_split=validation_split,
                    max_depth=self.data.get("max_depth") or 100.0,
                )
                if ckpt == plot_ckpt and provenance is not None:
                    self.depth_calibration = provenance
    except Exception as e:
        LOGGER.warning(f"Calibration skipped ({type(e).__name__}: {e}); checkpoints left uncalibrated.")
```

**[5] `DetectionValidator.eval_json`** (ultralytics/models/yolo/detect/val.py, lines 542–565)

```python
def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
    """Evaluate YOLO output in JSON format and return performance statistics.

    Args:
        stats (dict[str, Any]): Current statistics dictionary.

    Returns:
        (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
    """
    if self.gdict:
        predictions = iter(self.jdict)
        pred_json = [
            {**next(predictions), "image_id": image["id"]}
            for image, count in zip(self.gdict["images"], self.pred_counts)
            for _ in range(count)
        ]
    else:
        pred_json = self.jdict if self.training else self.save_dir / "predictions.json"
    anno_json = self.gdict or (
        self.data["path"]
        / "annotations"
        / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
    )
    return self.coco_evaluate(stats, pred_json, anno_json)
```

**[6] `BaseTrainer._do_train`** (ultralytics/engine/trainer.py, lines 540–605)

```python
        batch = self.preprocess_batch(batch)
        if self.args.compile:
            # Decouple inference and loss calculations for improved compile performance
            preds = self.model(batch["img"])
            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
        else:
            loss, self.loss_items = self.model(batch)
        self.loss = loss.sum()
        if RANK != -1:
            self.loss *= self.world_size
        if not self.loss_names:  # derive loss names from the criterion's loss dict on first batch
            self.loss_names = tuple(self.loss_items)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                self.metrics.update(dict.fromkeys(self.label_loss_items(prefix="val"), 0.0))
        self.tloss = (
            self.loss_items
            if self.tloss is None
            else {k: (self.tloss[k] * i + v) / (i + 1) for k, v in self.loss_items.items()}
        )

    # Backward
    self.scaler.scale(self.loss).backward()
except RuntimeError as e:
    is_oom = "out of memory" in str(e).lower()  # torch.cuda.OutOfMemoryError requires torch>=1.13
    if not is_oom and not any(
        s in str(e)
        for s in (
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDNN_STATUS_INTERNAL_ERROR",
            "unable to find an engine",
        )
    ):
        raise
    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
        raise  # only auto-reduce during first epoch on single GPU, max 3 retries
    self._oom_retries += 1
    old_batch = self.batch_size
    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
    error = f"{self.device.type.upper()} out of memory" if is_oom else "CUDA backend memory error"
    LOGGER.warning(
        f"{error} with batch={old_batch}. "
        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
    )
    batch = loss = preds = None
    self.loss = self.loss_items = self.tloss = None
    if hasattr(self.train_loader, "close"):
        self.train_loader.close()  # free the replaced loader's workers and prefetched batches
    self._clear_memory()
    self._build_train_pipeline()  # retain optimizer state across OOM retries
    mosaic_closed = not self.args.close_mosaic  # the rebuilt loader reopened mosaic, re-arm the gate
    self.validator.dataloader = self.test_loader  # the validator holds the pre-halving loader
    self.scheduler.last_epoch = self.start_epoch - 1
    nb = len(self.train_loader)
    nw = self._get_warmup_iterations(nb)
    last_opt_step = -1
    self.optimizer.zero_grad()
    break  # restart epoch loop with reduced batch size
if ni - last_opt_step >= self.accumulate:
    self.optimizer_step()
    last_opt_step = ni

    # Timed stopping
    if self.args.time:
        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
        if RANK != -1:  # if DDP training
```

</details>

## 10. `export-openvino`

**Question:** How do I export a YOLO model to OpenVINO, and can the export be quantized to INT8?

**Key facts:**

- [x] Use model.export(format='openvino'); it writes the OpenVINO .xml/.bin model and metadata.yaml into a *_openvino_model folder.
- [x] INT8 quantization is supported (quantize=8): NNCF calibrates the model with images from the data argument.

<details><summary>Sources</summary>

**[1] `Exporter.export_openvino`** (ultralytics/engine/exporter.py, lines 1247–1295)

```python
@try_export
def export_openvino(self, prefix=colorstr("OpenVINO:")):  # noqa: B008
    """Export YOLO model to OpenVINO format."""
    from ultralytics.utils.export.openvino import torch2openvino

    # OpenVINO <= 2025.1.0 error on macOS 15.4+: https://github.com/openvinotoolkit/openvino/issues/30023
    check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
    import openvino as ov

    assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"

    def serialize(ov_model, file):
        """Set RT info, serialize, and save metadata YAML."""
        ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
        ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
        ov_model.set_rt_info(114, ["model_info", "pad_value"])
        ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
        ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
        ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
        if self.model.task != "classify":
            ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

        ov.save_model(ov_model, file, compress_to_fp16=self.args.quantize == 16)
        YAML.save(Path(file).parent / "metadata.yaml", self.metadata)  # add metadata.yaml

    calibration_dataset = None
    if self.args.quantize == 8:
        check_requirements("packaging>=23.2")  # must be installed first to build nncf wheel
        check_requirements("nncf>=2.14.0,<3.0.0" if not TORCH_2_3 else "nncf>=2.14.0")
        import nncf

        calibration_dataset = nncf.Dataset(self.get_int8_calibration_dataloader(prefix), self._transform_fn)

    ov_model = torch2openvino(
        model=NMSModel(self.model, self.args) if self.args.nms else self.model,
        im=self.im,
        dynamic=self.args.dynamic,
        quantize=self.args.quantize,
        calibration_dataset=calibration_dataset,
        int8_detect=isinstance(self.model.model[-1], (Detect, RTDETRDecoder)),
        prefix=prefix,
    )

    suffix = f"_{'int8_' if self.args.quantize == 8 else ''}openvino_model{os.sep}"
    f = str(self.file).replace(self.file.suffix, suffix)
    f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

    serialize(ov_model, f_ov)
    return f
```

**[2] `ultralytics.engine.exporter`** (ultralytics/engine/exporter.py, lines 1–2032)

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLO PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo26n.pt
TorchScript             | `torchscript`             | yolo26n.torchscript
ONNX                    | `onnx`                    | yolo26n.onnx
OpenVINO                | `openvino`                | yolo26n_openvino_model/
TensorRT                | `engine`                  | yolo26n.engine
CoreML                  | `coreml`                  | yolo26n.mlpackage
Apple Core AI           | `coreai`                  | yolo26n.aimodel
TensorFlow SavedModel   | `saved_model`             | yolo26n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo26n.pb
TensorFlow Edge TPU     | `edgetpu`                 | yolo26n_edgetpu.tflite
LiteRT                  | `litert`                  | yolo26n.tflite
PaddlePaddle            | `paddle`                  | yolo26n_paddle_model/
MNN                     | `mnn`                     | yolo26n.mnn
NCNN                    | `ncnn`                    | yolo26n_ncnn_model/
IMX                     | `imx`                     | yolo26n_imx_model/
RKNN                    | `rknn`                    | yolo26n_rknn_model/
ExecuTorch              | `executorch`              | yolo26n_executorch_model/
Axelera AI              | `axelera`                 | yolo26n_axelera_model/
DEEPX                   | `deepx`                   | yolo26n_deepx_model/
Qualcomm QNN            | `qnn`                     | yolo26n_qnn.onnx
Hailo                   | `hailo`                   | yolo26n_hailo_model/
Huawei Ascend           | `ascend`                  | yolo26n_ascend_model/
AMD Xilinx              | `xilinx`                  | yolo26n_xilinx_model/

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolo26n.pt')
    results = model.export(format='onnx')
    results = model.export(format='onnx', quantize=8, data='coco8.yaml')  # INT8 ONNX

CLI:
    $ yolo export model=yolo26n.pt format=onnx
    $ yolo export model=yolo26n.pt format=onnx quantize=8 data=coco8.yaml

Inference:
    $ yolo predict model=yolo26n.pt                 # PyTorch
                         yolo26n.torchscript        # TorchScript
                         yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolo26n_openvino_model     # OpenVINO
                         yolo26n.engine             # TensorRT
                         yolo26n.mlpackage          # CoreML (macOS-only)
                         yolo26n.aimodel            # Apple Core AI (export on macOS 26+ Apple silicon or x86_64 Linux)
                         yolo26n_saved_model        # TensorFlow SavedModel
                         yolo26n.pb                 # TensorFlow GraphDef
                         yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo26n.tflite             # LiteRT
                         yolo26n_paddle_model       # PaddlePaddle
                         yolo26n.mnn                # MNN
                         yolo26n_ncnn_model         # NCNN
                         yolo26n_imx_model          # IMX
                         yolo26n_rknn_model         # RKNN
                         yolo26n_executorch_model   # ExecuTorch
                         yolo26n_axelera_model      # Axelera AI
                         yolo26n_deepx_model        # DEEPX
                         yolo26n_qnn.onnx           # Qualcomm QNN
                         yolo26n_hailo_model        # Hailo
                         yolo26n_ascend_model       # Huawei Ascend
                         yolo26n_xilinx_model       # AMD Xilinx
"""
from __future__ import annotations
import json
```

**[3] `Exporter`** (ultralytics/engine/exporter.py, lines 530–1868)

```python
    @try_export
    def export_onnx(self, prefix=colorstr('ONNX:')):
        """Export YOLO model to ONNX format."""
        ...
    @try_export
    def export_openvino(self, prefix=colorstr('OpenVINO:')):
        """Export YOLO model to OpenVINO format."""
        ...
    @try_export
    def export_paddle(self, prefix=colorstr('PaddlePaddle:')):
        """Export YOLO model to PaddlePaddle format."""
        ...
    @try_export
    def export_litert(self, prefix=colorstr('LiteRT:')):
        """Export YOLO model to LiteRT format using litert_torch with optional INT8 quantization."""
        ...
    @try_export
    def export_mnn(self, prefix=colorstr('MNN:')):
        """Export YOLO model to MNN format using MNN https://github.com/alibaba/MNN."""
        ...
    @try_export
    def export_ncnn(self, prefix=colorstr('NCNN:')):
        """Export YOLO model to NCNN format using PNNX https://github.com/pnnx/pnnx."""
        ...
    @try_export
    def export_coreml(self, prefix=colorstr('CoreML:')):
        """Export YOLO model to CoreML format."""
        ...
    @try_export
    def export_engine(self, prefix=colorstr('TensorRT:')):
        """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
        ...
    @try_export
    def export_saved_model(self, prefix=colorstr('TensorFlow SavedModel:')):
        """Export YOLO model to TensorFlow SavedModel format."""
        ...
    @try_export
    def export_pb(self, keras_model, prefix=colorstr('TensorFlow GraphDef:')):
        """Export YOLO model to TensorFlow GraphDef *.pb format https://github.com/leimao/Frozen-Graph-TensorFlow."""
        ...
    @try_export
    def export_axelera(self, prefix=colorstr('Axelera:')):
        """Export YOLO model to Axelera format."""
        ...
    @try_export
    def export_executorch(self, prefix=colorstr('ExecuTorch:')):
        """Export YOLO model to ExecuTorch *.pte format."""
        ...
    @try_export
    def export_coreai(self, prefix=colorstr('Core AI:')):
        """Export YOLO model to Apple Core AI *.aimodel format."""
        ...
    @try_export
    def export_edgetpu(self, tflite_model: Path, prefix=colorstr('Edge TPU:')):
        """Export YOLO model to Edge TPU format https://coral.ai/docs/edgetpu/models-intro/."""
        ...
    @try_export
    def export_rknn(self, prefix=colorstr('RKNN:')):
        """Export YOLO model to RKNN format with optional INT8 quantization."""
        ...
    @try_export
    def export_ascend(self, prefix=colorstr('Ascend:')):
        """Export YOLO model to Huawei Ascend offline model (.om) format."""
        ...
    @try_export
    def export_imx(self, prefix=colorstr('IMX:')):
        """Export YOLO model to IMX format."""
        ...
    @try_export
    def export_deepx(self, prefix=colorstr('DEEPX:')):
        """Export YOLO model to DEEPX format."""
        ...
    @try_export
    def export_qnn(self, prefix=colorstr('Qualcomm QNN:')):
        """Export YOLO model to a Qualcomm QNN context binary using ONNX Runtime QNN."""
        ...
    @try_export
    def export_xilinx(self, prefix=colorstr('AMD Xilinx:')):
        """Export YOLO model to AMD Xilinx Vitis AI format for Versal AI Edge Series Gen 2 NPUs."""
        ...
    @try_export
    def export_hailo(self, prefix=colorstr('Hailo:')):
        """Export a YOLO model to Hailo Executable Format (HEF)."""
        ...
    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://ai.google.dev/edge/litert/models/metadata."""
        ...
    def _transform_fn(self, data_item) -> np.ndarray:
        """Quantization preprocessing transform for INT8 calibration (Axelera, OpenVINO, ONNX, QNN)."""
        ...
    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        ...
    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        ...
```

**[4] `Exporter.export_litert`** (ultralytics/engine/exporter.py, lines 1310–1331)

```python
@try_export
def export_litert(self, prefix=colorstr("LiteRT:")):  # noqa: B008
    """Export YOLO model to LiteRT format using litert_torch with optional INT8 quantization.

    Supports ``quantize=8`` (static INT8, int8 weights + int8 activations, requires calibration ``data``),
    ``quantize='w8a16'`` (static, int8 weights + int16 activations, requires calibration ``data``) and
    ``quantize='w8a32'`` (dynamic/weight-only INT8, int8 weights + FP32 activations, no calibration needed).
    """
    assert MACOS or (LINUX and not ARM64), "LiteRT export only supported on Linux x86 and macOS"
    from ultralytics.utils.export.litert import torch2litert

    return torch2litert(
        self.model,
        self.im,
        self.file,
        quantize=self.args.quantize,
        calibration_dataset=self.get_int8_calibration_dataloader(prefix)
        if self.args.quantize in {8, "w8a16"}
        else None,
        metadata=self.metadata,
        prefix=prefix,
    )
```

**[5] `Model.export`** (ultralytics/engine/model.py, lines 753–805)

```python
def export(
    self,
    **kwargs: Any,
) -> str:
    """Export the model to a different format suitable for deployment.

    This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
    purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
    defaults, and any additional arguments provided.

    Args:
        **kwargs (Any): Arbitrary keyword arguments for export configuration. Common options include:
            - format (str): Export format (e.g., 'onnx', 'engine', 'coreml').
            - quantize (int | str): Precision, e.g. 16 (FP16) or 8 (INT8); 32/None is FP32.
            - device (str): Device to run the export on.
            - workspace (int): Maximum memory workspace size for TensorRT engines.
            - nms (bool | None): None for raw output, True to embed NMS, or False for the NMS-free head where available.
            - simplify (bool): Simplify ONNX model.

    Returns:
        (str): The path to the exported model file.

    Raises:
        TypeError: If the model is not a PyTorch model.
        ValueError: If an unsupported export format is specified.
        RuntimeError: If the export process fails due to errors.

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> model.export(format="onnx", dynamic=True, simplify=True)
        'path/to/exported/model.onnx'
    """
    self._check_is_pytorch_model()
    from .exporter import Exporter, export_formats

    custom = {
        "imgsz": self.model.args["imgsz"],
        "batch": 1,
        "data": None,
        "device": None,  # reset to avoid multi-GPU errors
        "verbose": False,
    }  # method defaults
    args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
    try:
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)
    except Exception:
        formats = export_formats()
        export_format = args.get("format", DEFAULT_CFG_DICT["format"])
        format_name = dict(zip(formats["Argument"], formats["Format"])).get(
            str(export_format).lower(), export_format
        )
        LOGGER.info(f"Export to {format_name} in the cloud with Ultralytics Platform: {PLATFORM_URL}")
        raise
```

**[6] `Exporter.export_engine`** (ultralytics/engine/exporter.py, lines 1437–1460)

```python
@try_export
def export_engine(self, prefix=colorstr("TensorRT:")):  # noqa: B008
    """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
    assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
    f_onnx = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016
    from ultralytics.utils.export.engine import onnx2engine

    assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
    f = self.file.with_suffix(".engine")  # TensorRT engine file
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

</details>
