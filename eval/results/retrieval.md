# Retrieval evaluation

_Generated on 2026-10-10 by `uv run code-assistant eval retrieval`._

- **Corpus:** Ultralytics `v8.4.175` (`69c95d1`), 104 files, 1532 chunks.
- **Questions:** 40 in scope (from real GitHub issues, hand-labelled before running any model; 14 name a symbol or argument), plus 6 out-of-scope questions not scored here.
- **Search:** exact cosine over cached vectors; parts of the same symbol are collapsed, so rankings contain distinct code units.
- **Metrics:** Hit@k = share of questions with at least one relevant symbol in the top k. MRR@10 = mean of 1/rank of the first relevant symbol (0 if not in the top 10). 95% CI from a 10,000-sample bootstrap over questions.
- **Query latency:** median time to embed one question on CPU.

## Results

| Model | Params | Dim | Hit@1 | Hit@5 (95% CI) | MRR@10 | Query latency (CPU) |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 596M | 1024 | 47.5% | **80.0%** (67.5%–92.5%) | 0.608 | 764 ms |
| `BAAI/bge-small-en-v1.5` ✅ | 33M | 384 | 52.5% | **77.5%** (65.0%–90.0%) | 0.626 | 117 ms |
| `ibm-granite/granite-embedding-english-r2` | 149M | 768 | 47.5% | **77.5%** (65.0%–90.0%) | 0.607 | 188 ms |
| `Alibaba-NLP/gte-modernbert-base` | 149M | 768 | 45.0% | **72.5%** (57.5%–85.0%) | 0.574 | 173 ms |

## Decision

Rule (fixed before running the evaluation): pick the best Hit@5; if a smaller model is within 3% of it, pick the smaller model.

**Chosen: `BAAI/bge-small-en-v1.5`.** Qwen/Qwen3-Embedding-0.6B has the best Hit@5 (80.0%), but BAAI/bge-small-en-v1.5 is within 3% (77.5%) with 33M instead of 596M parameters, so the smaller model wins.

With 40 questions, one question is worth 2.5%, and the confidence intervals above overlap. Small differences between models are not statistically meaningful, which is why the rule prefers the cheaper model on near-ties.

## Hit@5 by category

| Model | data (n=11) | export (n=4) | models (n=4) | prediction (n=7) | training (n=12) | validation (n=2) |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 100.0% | 100.0% | 75.0% | 57.1% | 83.3% | 0.0% |
| `ibm-granite/granite-embedding-english-r2` | 81.8% | 100.0% | 75.0% | 71.4% | 75.0% | 50.0% |
| `Alibaba-NLP/gte-modernbert-base` | 81.8% | 100.0% | 50.0% | 57.1% | 83.3% | 0.0% |
| `BAAI/bge-small-en-v1.5` | 81.8% | 100.0% | 75.0% | 57.1% | 75.0% | 100.0% |

## Hit@5 by question type

| Model | Names a symbol | Plain language |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 85.7% (n=14) | 76.9% (n=26) |
| `ibm-granite/granite-embedding-english-r2` | 100.0% (n=14) | 65.4% (n=26) |
| `Alibaba-NLP/gte-modernbert-base` | 92.9% (n=14) | 61.5% (n=26) |
| `BAAI/bge-small-en-v1.5` | 100.0% (n=14) | 65.4% (n=26) |

## Atlas approximate search vs exact search

For the deployed model (`BAAI/bge-small-en-v1.5`), Atlas `$vectorSearch` (numCandidates = 20 × limit) was run on the same questions.

- Mean overlap of the top-5 with the exact search: **99.0%**
- Hit@5 exact: 77.5% · Hit@5 Atlas: 77.5%

## Failure analysis (`BAAI/bge-small-en-v1.5`)

9 of 40 questions have no relevant symbol in the top 5.

### `train-freeze` (not in top 10)

> How can I freeze the backbone layers when doing transfer learning on a new dataset?

- Expected: `engine/trainer.py::BaseTrainer._setup_train`
- Retrieved: `models/sam/modules/sam.py::SAM2Model._prepare_backbone_features`, `models/sam/modules/tiny_encoder.py::TinyViT.set_layer_lr_decay`, `models/sam/sam3/encoder.py::TransformerEncoder.__init__`

### `train-grad-accum` (not in top 10)

> Does YOLO accumulate gradients so I can simulate a larger batch size on a small GPU?

- Expected: `engine/trainer.py::BaseTrainer._build_train_pipeline`, `engine/trainer.py::BaseTrainer._do_train`, `engine/trainer.py::BaseTrainer.optimizer_step`
- Retrieved: `models/yolo/detect/train.py::DetectionTrainer.preprocess_batch`, `models/yolo/detect/train.py::DetectionTrainer.auto_batch`, `models/yolo/yoloe/predict.py::YOLOEVPDetectPredictor`

### `train-plot-augmented` (not in top 10)

> How can I see what my training images look like after augmentation?

- Expected: `models/yolo/detect/train.py::DetectionTrainer.plot_training_samples`, `engine/trainer.py::BaseTrainer.plot_training_samples`
- Retrieved: `data/augment.py::LetterBox.__init__`, `data/augment.py::ultralytics.data.augment`, `data/augment.py::classify_augmentations`

### `data-multi-rtsp` (not in top 10)

> How can I run inference on several RTSP camera streams at the same time?

- Expected: `data/loaders.py::LoadStreams`, `data/loaders.py::LoadStreams.__init__`
- Retrieved: `engine/predictor.py::BasePredictor.stream_inference`, `data/build.py::load_inference_source`, `models/sam/predict.py::SAM3VideoSemanticPredictor.inference`

### `data-rtsp-reconnect` (rank 10)

> What happens when an RTSP stream loses its connection or returns empty frames during prediction?

- Expected: `data/loaders.py::LoadStreams.update`
- Retrieved: `engine/predictor.py::ultralytics.engine.predictor`, `engine/predictor.py::BasePredictor.stream_inference`, `models/sam/predict.py::SAM2VideoPredictor._get_empty_mask_ptr`

### `predict-returns-list` (rank 6)

> I get AttributeError: 'list' object has no attribute 'boxes' after calling the model. What does prediction return?

- Expected: `engine/model.py::Model.predict`, `engine/model.py::Model.__call__`, `engine/results.py::Results`
- Retrieved: `models/rtdetr/predict.py::RTDETRPredictor.postprocess`, `models/yolo/segment/predict.py::SegmentationPredictor.construct_result`, `engine/results.py::Boxes.cls`

### `predict-box-size` (not in top 10)

> How do I get the width and height in pixels of each detected object?

- Expected: `engine/results.py::Boxes.xywh`, `engine/results.py::Boxes`
- Retrieved: `models/yolo/detect/val.py::DetectionValidator.gather_stats`, `models/yolo/detect/val.py::DetectionValidator._gather_image_metrics`, `engine/results.py::Masks.xy`

### `predict-plot-no-boxes` (not in top 10)

> How can I draw only the segmentation masks on the output image, without the bounding boxes?

- Expected: `engine/results.py::Results.plot`
- Retrieved: `models/sam/amg.py::batched_mask_to_box`, `models/yolo/segment/predict.py::SegmentationPredictor.construct_result`, `engine/results.py::Masks`

### `model-sam2-dynamic` (not in top 10)

> How can I add new objects to track while running SAM2 interactively on a video?

- Expected: `models/sam/predict.py::SAM2DynamicInteractivePredictor`, `models/sam/predict.py::SAM2DynamicInteractivePredictor.inference`
- Retrieved: `models/sam/predict.py::SAM3VideoSemanticPredictor._tracker_add_new_objects`, `models/sam/modules/sam.py::SAM2Model._track_step`, `models/sam/modules/sam.py::SAM2Model.track_step`

## Questions every model misses

`train-grad-accum`, `train-plot-augmented`, `predict-returns-list`, `predict-plot-no-boxes`
