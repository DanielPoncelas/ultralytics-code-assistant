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

**Ask a question** (retrieval + LLM answer, streamed, with cited sources):

```bash
uv run code-assistant ask "How do I resume an interrupted training run?"
```

The answer cites sources as `[n]`. The command then lists the retrieved code units with GitHub permalinks (cited ones are marked `*`) and says which LLM answered, including any model that failed before it.

**Search the code** (retrieval only, no LLM):

```bash
uv run code-assistant search "How do I resume an interrupted training run?"
uv run code-assistant search "how are masks built from the network output" --task segment -k 3 --show-code
```

Each result shows the symbol, file and lines, and a permalink to the exact lines on GitHub at the indexed commit. Filters: `--folder {models,engine,data}`, `--task {detect,segment,pose,obb,classify,...}`, `--kind {function,method,class,module}`.

**Evaluate** retrieval and answers, and reproduce the model choices:

```bash
uv run code-assistant eval retrieval     # embedding models → eval/results/retrieval.md
uv run code-assistant eval answers       # LLMs + judge       → eval/results/answers.md
uv run code-assistant eval review        # blind human check of the judge (then: eval agreement)
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
| `LLM_MODELS` | no | see `config.py` | Comma-separated OpenRouter models, tried in order (fallback chain) |
| `RETRIEVAL_K` | no | `6` | Code units given to the LLM |
| `MONGODB_DATABASE` / `MONGODB_COLLECTION` | no | `code_assistant` / `chunks` | Where chunks are stored |

## Example questions

Five questions from real Ultralytics GitHub issues, chosen before generating any answer. They cover training, data, validation, export, and one question the index cannot answer. The full answers, unedited, with their cited sources, are in **[docs/examples.md](docs/examples.md)** (generated with `uv run code-assistant examples`, default configuration).

| # | Question (source issue) | What the assistant answered | Verdict |
|---|---|---|---|
| 1 | *My training run was interrupted at epoch 63. How do I resume it, and what state gets restored?* ([#17492](https://github.com/ultralytics/ultralytics/issues/17492)) | Resume with `resume=True`. Walks through `Model.train` → `check_resume` → `resume_training` (restarts at epoch 64), and gives a table of the checkpoint keys that are restored (optimizer, scaler, EMA, best fitness, …). | ✅ Correct and well cited |
| 2 | *Which data augmentations are applied by default when training a detection model?* ([#2412](https://github.com/ultralytics/ultralytics/issues/2412)) | `v8_transforms`: Mosaic, RandomPerspective, CopyPaste, MixUp, CutMix, Albumentations, RandomHSV and flips, each controlled by a hyperparameter. | ✅ Correct. Minor slip: it presents Albumentations as user-provided only |
| 3 | *Why do `model.val()` and `model.predict()` give different detections?* ([#2783](https://github.com/ultralytics/ultralytics/issues/2783)) | Different default arguments: `predict` uses `conf=0.25` and `rect=True`; the validator uses its own, lower `conf`. Pass the same arguments to both. | 🟡 Partly. The validator's `conf=0.001` wasn't retrieved, so the answer says it's "implied" rather than stating it, which is honest about the gap |
| 4 | *How can I export to ONNX with a dynamic batch size and NMS in the graph?* ([#23339](https://github.com/ultralytics/ultralytics/issues/23339)) | `model.export(format="onnx", dynamic=True, nms=True, batch=16)`, with `batch` as the maximum batch size. | ✅ Correct, but it cites no source |
| 5 | *What are the default values of the initial learning rate and momentum?* ([#2849](https://github.com/ultralytics/ultralytics/issues/2849)) | "lr0 = 1e-5, momentum = 0.7", read from the hyperparameter-tuning search space. | ❌ Wrong. The defaults live in `cfg/default.yaml`, which isn't indexed; the model presented the tuner's lower bounds as defaults instead of saying it couldn't find them. A known failure mode, see [LLM choice](#llm-choice) |

## Design notes

### Pipeline

```
Indexing (one-off)   fetch (git, pinned tag + commit check) ─► chunk (stdlib ast) ─► embed (local open-source model) ─► store (Atlas, upsert by id)

Answering            question ─► embed ─► Atlas $vectorSearch (top 6 distinct code units) ─► prompt with numbered sources ─► free LLM via OpenRouter (fallback chain) ─► streamed answer with [n] citations + GitHub permalinks
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

### Answer generation

- **Prompt:**
  - the sources are numbered and wrapped in `<source>` tags;
  - the model answers only from them, cites `[n]` per statement, answers the supported part and says what is missing, and replies "I couldn't find this in the indexed code" only when nothing relevant was retrieved;
  - retrieved text is treated as untrusted data (indirect prompt-injection guard, tested).
- **Context:** 6 code units, the knee of the retrieval curve (Hit@5 77.5%, Hit@6 80%, Hit@8 80%).
- **Fallback chain:** if the first model is rate-limited, errors, times out or returns an empty answer, the next one answers. The CLI says which model answered and which failed. While streaming, fallback only happens before the first token, so two models' answers are never spliced.

### LLM choice

The assignment asks for a free OpenRouter model. On 2026-10-10 OpenRouter listed 15 free models, and **none of them were Qwen, Llama, DeepSeek or Mistral** (all paid). From those 15 we excluded:
- the Nemotron family, because it's the judge's family;
- Inkling, which isn't available through the API;
- a 2.6B model, too small.

The candidates were two code-specialised models (Cohere, Poolside) and one general model (Google Gemma 4).

**Method:**
- 46 questions from real issues; 40 have 2–3 hand-written *key facts* each, checked against the source.
- Every model answered from the **same retrieved sources**.
- A judge from a different model family (`nvidia/nemotron-3-super-120b-a12b:free`) graded key-fact coverage, unsupported claims and refusals.
- The selection rule was fixed in advance. Full report: [`eval/results/answers.md`](eval/results/answers.md).
- **Judge validation:** in a blind review by the author (10 random answers, the judge's verdicts not shown; [`eval/results/judge_review.md`](eval/results/judge_review.md)), the judge agreed with the human on **25/25 key facts** (Cohen's κ = 1.0; 95% CI for the agreement rate ≈ 86–100%). This covers key-fact grading only. One known judge error is in refusal detection: it missed a correct "couldn't find it" (see DECISIONS D23).

| Model | Key-fact coverage | Fully correct | Grounded | Out-of-scope declined | False refusals | Median latency |
|---|---|---|---|---|---|---|
| **`cohere/north-mini-code:free`** ✅ | **53.8%** | **32.5%** | 77.8% | 3/6 | **0%** | 10.1 s |
| `dots-studio/dots-3-note-preview:free` | 42.9% | 20.0% | **82.2%** | 3/6 | **0%** | 15.9 s |
| `poolside/laguna-s-2.1:free` | 36.2% | 12.5% | 80.4% | 4/6 | 18.8% | 8.5 s |
| `google/gemma-4-31b-it:free` | – | – | – | – | – | unavailable: rate-limited upstream for 37/46 questions |

**Chosen: Cohere North-Mini-Code.** It has the best coverage by 11 points, never declines when the right code is in context, and is reasonably fast. The default fallback chain is **Cohere → dots-3 → Laguna**: three providers, so one provider's outage doesn't take the assistant down.

**Honest limitations:**
- **Even the best free model states only about half of the key facts.**
- **It cites a source in only 45% of its answers.**
- **It sometimes answers out-of-scope questions with a plausible misreading instead of declining.** Example: tuner search-space bounds presented as default hyperparameters (example 5 above).
- **Free models are non-deterministic and their availability changes by the hour.**
- **Reasoning models (Cohere, dots-3) spend part of the token budget thinking.** The evaluation ran with 2,000 output tokens, which truncated 3 of Cohere's 46 answers. The default is now 4,000 and truncated answers are flagged.

Running the answer evaluation needs ~92 OpenRouter requests per model (answer + judge). Accounts with fewer than 10 credits purchased are limited to 50 free-model requests per day; we used an account with credits (1,000/day; free models don't consume them). All answers and verdicts are committed in `eval/results/answers.jsonl`.

## Development

```bash
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
```

Design decisions, alternatives considered and known limitations are logged in [DECISIONS.md](DECISIONS.md).

## Future work

_TODO: improvements, scaling to production, critical missing features._
