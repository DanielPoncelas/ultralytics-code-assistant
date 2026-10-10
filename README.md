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

   This clones Ultralytics at the pinned release (`v8.4.175`) and splits `ultralytics/{models,engine,data}` into ~1,500 chunks. It embeds them locally with an open-source model, uploads them to Atlas and creates the vector search index. The first run downloads the embedding model (~1.2 GB) from Hugging Face. The command is idempotent: re-running it updates chunks in place and never duplicates them. Computed embeddings are cached in `.cache/embeddings/`, so a second run skips the slow step.

   **Hardware and speed.** Embedding runs on the fastest device available, detected automatically:

   | Machine | Device used | Install | Approx. time (default model) |
   |---|---|---|---|
   | Mac with Apple Silicon | Apple GPU (MPS) | default `uv sync` | not benchmarked yet (no Mac available to the author) |
   | Linux/Windows + NVIDIA GPU | CUDA | `uv sync --no-group cpu --group cuda`, then run with `uv run --no-group cpu --group cuda code-assistant index` | ~27 min on a 4 GB GTX 1650 Ti (measured), much less on recent GPUs |
   | CPU only | CPU | default `uv sync` | hours for the default model. Use a smaller model (`--model ibm-granite/granite-embedding-english-r2`) |

   Answering questions only embeds one short query (~0.6 s on a laptop CPU with the default model, measured), so a GPU only matters for this one-off step.

## Usage

_TODO_

## Example questions

_TODO: 3–5 questions from real YOLO use cases._

## Design notes

### Pipeline

```
 fetch (git, pinned tag + commit check) ─► chunk (stdlib ast) ─► embed (local model, CPU) ─► store (Atlas, upsert by id)
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

## Development

```bash
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
```

Design decisions, alternatives considered and known limitations are logged in [DECISIONS.md](DECISIONS.md).

## Future work

_TODO: improvements, scaling to production, critical missing features._
