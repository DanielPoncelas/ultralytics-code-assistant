# Ultralytics Code Assistant

A retrieval-augmented code assistant that answers questions about [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) by retrieving relevant source code and generating grounded answers.

> 🚧 Work in progress.

## Prerequisites

| Tool | Version | Why | Install |
|------|---------|-----|---------|
| [uv](https://docs.astral.sh/uv/) | ≥ 0.13.0 | Package and Python version manager (required by the assignment) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [git](https://git-scm.com/downloads) | any recent | Clones this repo and the pinned Ultralytics source | Preinstalled on most systems |

**Hardware:** any machine with ~4 GB free RAM and ~4 GB free disk (Python environment and embedding model). **No GPU needed**: everything runs on CPU.

> You do **not** need to install Python yourself: uv downloads the exact interpreter version pinned in `.python-version`.

### Accounts

| Service | Tier | Why | What you need from it |
|---------|------|-----|-----------------------|
| [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) | Free (M0) | Vector store for code chunks (required by the assignment) | A connection string (`mongodb+srv://...`) |
| [OpenRouter](https://openrouter.ai) | Free | LLM API used to generate answers (required by the assignment) | An API key (`sk-or-v1-...`) from **Keys → Create API Key** |

> **OpenRouter free-tier limit:** accounts without credits can make **50 requests/day** to `:free` models, which is enough to try the app. Running the full evaluation suite needs more.

<details>
<summary>MongoDB Atlas setup (≈5 min)</summary>

1. Create an account and choose **Create Deployment → Free**. Any provider or region works (we used AWS `eu-west-3`, Paris). Leave "Preload sample dataset" unchecked.
2. Keep **Automate security setup** enabled. Atlas adds your current IP to the access list and creates a database user. Save the username and password.
3. Go to **Connect → Drivers → Python** and copy the connection string. Ignore the `pip install` hint, because dependencies are installed with `uv`.

You don't need to create the vector search index by hand. The indexing command creates it.
</details>

## Setup

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/DanielPoncelas/ultralytics-code-assistant.git
   cd ultralytics-code-assistant
   ```

2. Create your environment file and fill in the two values obtained in [Accounts](#accounts):

   ```bash
   cp .env.example .env
   ```

3. Install Python 3.12 and all dependencies at the exact versions pinned in `uv.lock`:

   ```bash
   uv sync
   ```

4. Verify configuration and connectivity:

   ```bash
   uv run code-assistant check
   ```

   Expected output:

   ```
   ✔ Settings loaded
   ✔ MongoDB Atlas reachable (server 8.0.x)
   ✔ OpenRouter key valid (free-model requests today: 50/50)
   ```

   If a line shows `✘`, the message explains which setting or service to fix.

5. Build the index (one-off):

   ```bash
   uv run code-assistant index
   ```

   This clones Ultralytics at the pinned release (`v8.4.175`) and splits `ultralytics/{models,engine,data}` into ~1,500 chunks. It embeds them locally with an open-source model, uploads them to Atlas and creates the vector search index. The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) from Hugging Face. The command is idempotent: re-running it updates chunks in place and never duplicates them. Computed embeddings are cached in `.cache/embeddings/`, so a second run skips the slow step.

   **Hardware and speed.** Embedding runs on the fastest device available (NVIDIA GPU, Apple GPU, otherwise CPU), detected automatically. No GPU is required with the default model.

   *Reference timings for the 1,532 chunks with the default model, measured on the author's laptop. Times on your machine will differ; they are given for order of magnitude only:*

   | Device | Indexing time |
   |---|---|
   | CPU: Intel Core i7-10750H (6 cores / 12 threads) | 395 s (~6.6 min), peak RAM ~1 GB |
   | GPU: NVIDIA GTX 1650 Ti, 4 GB (install with `uv sync --no-group cpu --group cuda`, run with `uv run --no-group cpu --group cuda code-assistant index`) | 104 s |
   | Apple Silicon GPU (automatic on Mac) | not measured (no Mac available to the author) |

   Larger candidate models can be indexed with `--model` (see [Embedding model choice](#embedding-model-choice)). In practice those need a GPU: on the same laptop, Qwen3-Embedding-0.6B took ~27 min on the GTX 1650 Ti and had a 9+ hour ETA on the CPU.

## Usage

**Search the code** (retrieval only, no LLM):

```bash
uv run code-assistant search "How do I resume an interrupted training run?"
uv run code-assistant search "how are masks built from the network output" --task segment -k 3 --show-code
```

Each result shows the symbol, file and lines, and a permalink to the exact lines on GitHub at the indexed commit. Filters: `--folder {models,engine,data}`, `--task {detect,segment,pose,obb,classify,...}`, `--kind {function,method,class,module}`.

**Evaluate retrieval** and reproduce the embedding model choice:

```bash
uv run code-assistant eval retrieval     # writes eval/results/retrieval.md and .json
```

Evaluating all four candidate models needs their embeddings. The first run computes them, which takes minutes on a GPU and hours on CPU for the larger models. Use `--model BAAI/bge-small-en-v1.5` to evaluate only the default model.

### Configuration

All settings are read from environment variables or `.env`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MONGODB_URI` | yes | – | Atlas connection string |
| `OPENROUTER_API_KEY` | yes | – | LLM API key |
| `EMBEDDING_MODEL` | no | `BAAI/bge-small-en-v1.5` | One of the pinned candidates in `embeddings.py` |
| `EMBEDDING_DEVICE` | no | auto | `cuda`, `mps` or `cpu` |
| `MONGODB_DATABASE` / `MONGODB_COLLECTION` | no | `code_assistant` / `chunks` | Where chunks are stored |

## Example questions

Five questions taken from real Ultralytics GitHub issues. They were chosen before generating any answer and cover training, data, validation, export, plus one question the index can't answer. Below are the code units retrieved for each one (top 3, default model). Generated answers are added with the answer-generation step.

| # | Question (source issue) | Top retrieved code | Expected code found? |
|---|---|---|---|
| 1 | *My training run was interrupted at epoch 63. How do I resume it from the last checkpoint, and what state gets restored?* ([#17492](https://github.com/ultralytics/ultralytics/issues/17492)) | [`BaseTrainer.resume_training`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L1124-L1141), [`BaseTrainer.check_resume`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L1020-L1065), [`Model.train`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L875-L913) | ✅ rank 1 |
| 2 | *Which data augmentations are applied by default when training a detection model?* ([#2412](https://github.com/ultralytics/ultralytics/issues/2412)) | [`Model.train`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L807-L877), [`DetectionTrainer`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/detect/train.py#L24-L256), [`v8_transforms`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/data/augment.py#L2831-L2912) | ✅ rank 3 |
| 3 | *Why do `model.val()` and `model.predict()` give different detections on the same images?* ([#2783](https://github.com/ultralytics/ultralytics/issues/2783)) | [`DetectionValidator.plot_predictions`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/detect/val.py#L451-L477), [`ClassificationValidator`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/classify/val.py#L18-L219), [`YOLOE.predict`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/yolo/model.py#L532-L567) | ❌ The answer is the different default `conf` (0.001 in `BaseValidator.__init__` vs 0.25 in `BasePredictor.__init__`), spread across files. A known failure, see the [evaluation report](eval/results/retrieval.md) |
| 4 | *How can I export a model to ONNX with a dynamic batch size and NMS included in the graph?* ([#23339](https://github.com/ultralytics/ultralytics/issues/23339)) | [`Exporter.export_onnx`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L1175-L1245), [`Exporter.__call__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/exporter.py#L842-L911), [`Model.export`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/model.py#L753-L805) | ✅ rank 1 |
| 5 | *What are the default values of the initial learning rate and momentum?* ([#2849](https://github.com/ultralytics/ultralytics/issues/2849)) | [`Tuner.__init__`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/tuner.py#L79-L142), [`BaseTrainer.build_optimizer`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/engine/trainer.py#L1215-L1242), [`TinyViT.set_layer_lr_decay`](https://github.com/ultralytics/ultralytics/blob/69c95d1add13a232382e53b3aa5094ec0c7738db/ultralytics/models/sam/modules/tiny_encoder.py#L900-L933) | ➖ Out of scope by design: defaults live in `ultralytics/cfg/default.yaml`, which isn't indexed. The assistant must say so instead of guessing |

## Design notes

### Pipeline

```
 fetch (git, pinned tag + commit check) ─► chunk (stdlib ast) ─► embed (local open-source model) ─► store (Atlas, upsert by id)
```

### How code is chunked, and why

Code is split by **logical unit using Python's `ast` module**, not by a fixed number of characters:

| Chunk type | Content | Answers questions like |
|---|---|---|
| **Function / method** | Full definition: decorators, signature, docstring, body. Nested functions stay inside their parent. | "How does `BaseTrainer.save_model` decide what to store?" |
| **Class summary** | Class header, docstring, class attributes, and every method's signature plus the first line of its docstring. No method bodies. | "What does `DetectionTrainer` do and which methods does it override?" |
| **Module summary** | Module docstring, imports, constants, `__all__`, and the signature of every top-level function and class. | "Where is `YOLODataset` defined / what does `ultralytics.data.augment` contain?" |

Why:
- **Fixed-size windows** cut functions in half and mix unrelated code, so neither the embedding nor the LLM sees a coherent unit.
- **Whole files** are too coarse. `exporter.py` has 1,500+ lines, and a single vector cannot represent it.
- **AST units** match how people ask about code: by function, method or class. The summaries cover the "big picture" questions that a single method can't answer.
- `ast` (stdlib) is enough because the corpus is pure Python. tree-sitter would only pay off for multiple languages.

**Long units.** Units over ~4,000 characters (~1,000–1,300 tokens; about 5% of them, e.g. the 444-line `Exporter.__call__`) are split **at statement boundaries**, never mid-statement, with a 3-line overlap. Each part is labelled `Part i/n`.

**Contextual header.** The text that is embedded is prefixed with where the code lives:

```
# File: ultralytics/engine/trainer.py
# Module: ultralytics.engine.trainer
# Symbol: BaseTrainer._do_train (method of class BaseTrainer)
# Signature: def _do_train(self)
# Part: 2/4
<code>
```

An isolated method body is ambiguous (`self.model`: which model?). The header places it in the codebase. The raw code is stored separately for display and for the LLM prompt.

**Comments.** `ast` doesn't see comments, so comment blocks between statements (e.g. `# NOTE: keep this class for backward compatibility` above a class) are attached to the statement that follows them. A coverage check over the whole corpus confirms that **every non-blank line of the 104 files appears in some chunk**. The only exceptions are the license headers of 4 `__init__.py` files that contain no code at all.

Result for `v8.4.175`: **104 files → 1,532 chunks** (1,072 methods, 214 class summaries, 140 functions, 106 module summaries; 79 long units split into parts). Median chunk 927 characters of code, max 4,000.

### Metadata extracted per chunk

| Field | Used for |
|---|---|
| `path`, `module`, `qualified_name`, `kind` | Identifying the chunk and citing it |
| `start_line`, `end_line`, `github_url` | Clickable citations: a permalink to the exact lines **at the pinned commit**, so links never rot |
| `signature`, `docstring`, `decorators`, `parent_class`, `bases` | Context for the LLM; building class summaries |
| `folder` (`models`/`engine`/`data`), `model_family` (`yolo`, `sam`, `rtdetr`, …), `task` (`detect`, `segment`, `pose`, …) | **Pre-filters** in Atlas Vector Search, e.g. "search only `task: segment`" |
| `part`, `total_parts` | Reassembling split units |
| `source.{repo,tag,commit}`, `embedding_model` | Provenance: which code version and which model produced each vector |
| `_id` | Stable hash of path + symbol + part, which makes re-indexing idempotent |

### Retrieval

1. Embed the question with the same model as the index. This is checked: querying an index built with another model fails with a clear error.
2. Run Atlas `$vectorSearch` with `limit = 3k` and `numCandidates = 20 × limit`. That's MongoDB's guidance for near-exact recall, and we **measured 99% top-5 overlap with an exact search**.
3. Collapse the parts of the same symbol (a long function split into parts could otherwise fill the top-k) and return k distinct code units.
4. Optional metadata pre-filters (`folder`, `task`, `kind`) narrow the search before ranking.

### Embedding model choice

All candidates are open source (Apache-2.0/MIT), not gated, natively supported by `transformers` (no remote code), and pinned to an exact revision:

| Candidate | Why it's in the comparison |
|---|---|
| `Qwen/Qwen3-Embedding-0.6B` (596M, 32k ctx) | The strongest model that fits on a laptop (top of MTEB for its size) |
| `ibm-granite/granite-embedding-english-r2` (149M, 8k ctx) | Mid-size ModernBERT model trained with code retrieval data |
| `Alibaba-NLP/gte-modernbert-base` (149M, 8k ctx) | Same size and architecture, different training data |
| `BAAI/bge-small-en-v1.5` (33M, 512 ctx) | Deliberately small, generic baseline |

Ruled out:
- `jina-code-embeddings` (non-commercial license);
- `embeddinggemma` (gated);
- `CodeRankEmbed` (its remote code crashes with transformers 5).

**How the model was chosen.** On 40 real questions labelled before any run, with this decision rule fixed in advance: *best Hit@5 wins, unless a smaller model is within 3 points*.

| Model | Hit@1 | Hit@5 (95% CI) | MRR@10 | Query time (CPU) |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 47.5% | **80.0%** (67.5–92.5) | 0.608 | 822 ms |
| **bge-small-en-v1.5** ✅ | **52.5%** | 77.5% (65.0–90.0) | **0.626** | **119 ms** |
| Granite Embedding R2 | 47.5% | 77.5% (65.0–90.0) | 0.607 | 199 ms |
| GTE-ModernBERT | 45.0% | 72.5% (57.5–85.0) | 0.574 | 189 ms |

**Chosen: `BAAI/bge-small-en-v1.5`.** It's one question behind the best model on Hit@5 and ahead on Hit@1 and MRR. It's also 18× smaller, embeds a query 7× faster on CPU, and makes indexing from scratch practical on any laptop. Trade-offs:
- **Truncation:** it reads at most 512 tokens, so 32.5% of chunks are truncated when embedded. The LLM still receives the full code.
- **Plain-language questions:** Qwen does better on questions that name no symbol (76.9% vs 65.4%, n=26). That's not significant at this sample size, but it's worth re-checking on a larger set.

Full report with per-category results and failure analysis: [`eval/results/retrieval.md`](eval/results/retrieval.md). Detailed reasoning: [DECISIONS.md](DECISIONS.md) D12–D18.

## Development

```bash
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
```

Design decisions, alternatives considered and known limitations are logged in [DECISIONS.md](DECISIONS.md).

## Future work

_TODO: improvements, scaling to production, critical missing features._
