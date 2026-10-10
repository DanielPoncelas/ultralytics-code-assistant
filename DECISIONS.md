# Decision log

Every significant design decision, with the alternatives considered and the trade-offs accepted. Entries are added as the project evolves; superseded decisions are marked, not deleted.

---

## D1. Python 3.12

- **Decision:** pin Python 3.12 (`.python-version`, `requires-python = ">=3.12"`).
- **Alternatives:** 3.13 / 3.14.
- **Why:** ML libraries used later (PyTorch, sentence-transformers) ship wheels for 3.12 reliably; newer versions sometimes lag. uv installs the interpreter automatically, so pinning costs reviewers nothing.
- **Trade-off:** we forgo the latest language features. None of them matter for this project.

## D2. `uv` with a committed lockfile and a minimum uv version

- **Decision:** dependencies in `pyproject.toml`, exact resolved versions in a committed `uv.lock`, and `[tool.uv] required-version = ">=0.13.0"`.
- **Why:** reproducibility is a hard requirement ("someone can run your code by following your README"). `uv sync` reproduces the exact environment, and an older uv fails with a clear message instead of an obscure resolution error.
- **Trade-off:** none worth noting. uv is also mandated by the assignment.

## D3. `src/` layout, package `code_assistant`

- **Decision:** code lives in `src/code_assistant/` and is installed as a package with a `code-assistant` console script.
- **Alternatives:** flat layout (package at repo root).
- **Why:** with a flat layout, Python can import the local folder even if packaging is broken, so tests pass on your machine and fail on the reviewer's. `src/` forces tests to run against the *installed* package. It is the layout recommended by the PyPA packaging guide.
- **Naming:** `ultralytics_*` was avoided so the package is never confused with the real `ultralytics` package.

## D4. Configuration with `pydantic-settings`

- **Decision:** a single typed `Settings` class reads environment variables and `.env`. Credentials are `SecretStr`.
- **Alternatives:** `os.environ` + `python-dotenv` by hand.
- **Why:** validation at startup gives a clear error naming the missing variable, instead of a confusing failure deep inside a network call. `SecretStr` keeps credentials out of logs, reprs and tracebacks (covered by a test).
- **Trade-off:** one extra dependency. pydantic is already a transitive dependency of the stack.

## D5. CLI with Typer, starting with a `check` command

- **Decision:** `uv run code-assistant <command>`. The first command, `check`, validates settings and pings Atlas and OpenRouter.
- **Alternatives:** `argparse` (stdlib, more boilerplate); loose scripts.
- **Why:** one entry point that is documented by `--help` and scales to `index`, `ask`, `eval` and `app`. `check` turns the setup verification into something any reviewer can repeat, and doubles as a pre-flight check before a live demo.

## D6. Plain `httpx` for HTTP calls (no LLM SDK)

- **Decision:** call the OpenRouter REST API directly with `httpx`.
- **Alternatives:** the `openai` SDK pointed at OpenRouter's OpenAI-compatible endpoint.
- **Why:** full visibility into requests, errors and retries, which we need for model fallback (D7). It's one small, well-known dependency, and it avoids any doubt about the "no pre-built frameworks" constraint.
- **Trade-off:** we write a little more code (request and response handling) ourselves.

## D7. LLM client must support ordered fallback models

- **Decision (to implement in the generation phase):** configure an ordered list of free models. On a rate limit (429), provider error or empty completion, try the next one. Report which model answered.
- **Evidence (2026-10-10 setup test):**
  - `google/gemma-4-31b-it:free` returned **429 "temporarily rate-limited upstream"**. Free models share an upstream quota pool across all OpenRouter users.
  - `nvidia/nemotron-3-super-120b-a12b:free` returned an **empty answer**. It is a reasoning model and spent the whole token budget reasoning.
  - `poolside/laguna-s-2.1:free` answered correctly.
- **Why:** without fallback, the availability of a free model is outside our control, and a live demo becomes a gamble.
- **Also:** free-tier accounts are limited to **50 free-model requests/day**. That limit matters for running evaluations.

## D8. Index a pinned release (`v8.4.175`), verified by commit SHA

- **Decision:** shallow-clone the tag and check that `HEAD` matches the pinned commit `69c95d1`. Fail if it doesn't.
- **Alternatives:**
  - `main`: not reproducible, because Ultralytics releases almost daily (5 releases in the 10 days before this decision).
  - `pip install ultralytics==X`: pulls PyTorch, OpenCV and more just to read `.py` files, and the paths wouldn't match GitHub.
- **Why the SHA check:** a tag can be moved. The commit can't. Citations link to `/blob/<commit>/...`, so they stay valid forever.
- **Trade-off:** answers reflect `v8.4.175`, not today's `main`. Re-indexing a new release means changing two config values.

## D9. AST-based chunking (functions/methods + class and module summaries)

- **Decision:** one chunk per function or method, plus a summary chunk per class and per module. See the README → Design notes.
- **Alternatives:**
  - Fixed-size windows: cut functions in half.
  - Whole files: too coarse (up to 4,000 lines).
  - tree-sitter: multi-language, which is unnecessary for pure Python.
  - Methods only, with no summaries: "what does class X do" questions would have no good chunk.
- **Measured before deciding:** 1,011 methods (median 18 lines), 127 functions, 188 classes. 27 units are over ~1.5k tokens, the largest 444 lines.
- **Trade-offs:**
  - Class and module summaries duplicate some information (signatures appear in both the summary and the method chunk). That's intentional: they serve different questions.
  - Very large class docstrings push some class summaries into 2 parts.
- **Comments:** `ast` drops comments, so comment blocks between statements are attached to the *following* statement (the Python convention). The first full-corpus audit found 45 meaningful comment lines being lost (e.g. `# NOTE: keep this class for backward compatibility`). After the fix, every non-blank line of the corpus is in some chunk, which a test enforces.

## D10. Splitting long units at statement boundaries, 4,000-char budget, 3-line overlap

- **Decision:** greedy packing up to 4,000 characters. Each part ends right before the latest statement start inside the window, and continuation parts repeat 3 lines.
- **Why characters, not tokens:** the chunk boundaries must be **identical for every embedding model**, so that model comparisons only measure the model. Each model has its own tokenizer, so a character budget is model-independent. 4,000 chars ≈ 1,000–1,300 tokens of Python.
- **Why 4,000:** ~95% of functions and methods fit in one chunk. Smaller chunks give more precise retrieval, while larger ones dilute the embedding with unrelated logic.
- **Trade-off:** a continuation part loses the enclosing block's context (e.g. that it's inside a `for` loop). The header (symbol, signature, part i/n) and the overlap mitigate this.
- **Tested:** every part of a split function parses on its own, so no statement is ever cut.

## D11. Contextual header in the embedded text (not in the stored code)

- **Decision:** embed `# File / Module / Symbol / Signature / Part` + code. Store the raw code separately.
- **Why:** a method body alone lacks its location. The header injects the path, class and signature vocabulary that users mention in questions ("trainer", "exporter", "segment"). It's cheap and has no extra model calls, unlike LLM-generated chunk descriptions.
- **Alternative left for future work:** LLM-generated natural-language summaries per chunk ("contextual retrieval"). Better recall for vague questions, but ~1,500 LLM calls per index build.

## D12. Embeddings: local sentence-transformers, pinned revisions, 4 candidates, chosen with data

- **Decision:** the model is configurable among 4 pre-selected candidates, each pinned to an exact Hugging Face revision. Provisional default: `Qwen/Qwen3-Embedding-0.6B`.
- **Candidates and why:**
  - **Qwen3-Embedding-0.6B** (596M, Apache-2.0, 32k ctx): strong code retrieval results on MTEB for its size, and instruction-aware queries. We use a **task-specific instruction** instead of the default "web search" one, as Qwen recommends.
  - **ibm-granite/granite-embedding-english-r2** (149M, Apache-2.0, 8k ctx): ModernBERT architecture, trained with code retrieval data, 4× smaller than Qwen.
  - **Alibaba-NLP/gte-modernbert-base** (149M, Apache-2.0, 8k ctx): ModernBERT architecture, strong on the CoIR code retrieval benchmark.
  - **BAAI/bge-small-en-v1.5** (33M, MIT, 512 ctx): a generic, short-context baseline that quantifies what a larger, code-aware model adds.
- **Ruled out:**
  - `jina-code-embeddings-0.5b`: **CC-BY-NC**, non-commercial.
  - `embeddinggemma-300m`: gated, so a reviewer must accept terms.
  - `Qodo-Embed-1-1.5B`: custom licence and slow.
  - **`nomic-ai/CodeRankEmbed`** (initially a candidate): it needs `trust_remote_code`, and its 2024 modelling code **crashes with transformers 5** (`'NomicBertModel' object has no attribute 'get_extended_attention_mask'`). Remote code runs arbitrary downloaded Python (a security risk) and breaks when libraries move on. **Rule adopted: only models natively supported by `transformers`.**
- **Final choice is made with data** in the retrieval evaluation (Recall@5 / MRR on our own question set). **The rule is fixed in advance:** pick the best Recall@5. If a smaller model is within 3 points, pick the smaller one.
- **Asymmetric encoding:** queries get the model's instruction prefix (Qwen only; the others use none) and documents don't. Vectors are L2-normalized.

## D13. Hardware: auto-detect GPU, CPU-only PyTorch by default, optional CUDA (supersedes "CPU-only")

- **What changed and why:** the first full run of Qwen3-0.6B on a laptop CPU (Intel, 12 threads) **took 9 minutes for the first batch of 8 chunks**, with an ETA of over 9 hours. A controlled benchmark (32 random real chunks, batch 8) confirmed that the CPU is not viable for the larger models:

  | Model | Laptop CPU | GTX 1650 Ti (4 GB, fp16) |
  |---|---|---|
  | bge-small (33M) | 1,837 tok/s → ~7 min corpus | 8,343 tok/s → ~1.5 min |
  | Granite R2 / GTE-ModernBERT (149M) | not measured | ~1,090 tok/s → ~11 min |
  | Qwen3-0.6B (596M) | < 20 tok/s → 9+ h | 296 tok/s → ~32 min |

- **Full-corpus runs on the GTX 1650 Ti (1,532 chunks):** bge-small 104 s, Granite R2 548 s, GTE-ModernBERT 550 s, Qwen3-0.6B 1,640 s (~27 min). Upload to Atlas from cache: 45 s. A re-run with nothing changed: 16 s, 0 writes.
- **Query time is a different story:** embedding **one question** with Qwen3-0.6B on the laptop CPU takes **~0.6 s** (3 questions measured). Atlas `$vectorSearch` answers in ~90 ms once connected. The CPU is only a bottleneck for bulk indexing, not for answering.
- **Decision:**
  1. **Device auto-detection:** CUDA, then Apple MPS, then CPU, overridable with `EMBEDDING_DEVICE`. Float16 on CUDA (Qwen fits in 4 GB VRAM), float32 elsewhere.
  2. **PyTorch build through dependency groups:** the default `cpu` group installs CPU-only wheels on Linux/Windows (~200 MB) and the regular macOS wheels (which include MPS) on Mac. `--no-group cpu --group cuda` installs CUDA 12.6 wheels (driver ≥ 525, chosen over cu130/cu132 for compatibility with older drivers).
- **Why groups, not extras:** uv can make a *group* default, but not an *extra*. With extras, a plain `uv run` silently replaced the CPU build with the multi-GB PyPI/CUDA build (verified with `uv sync --dry-run`).
- **Trade-offs:**
  - GPU users type a longer command, and only for `index`. Answering questions embeds one short query, which takes milliseconds on CPU.
  - Vectors from GPU fp16 and CPU fp32 differ in the last decimals. That doesn't change rankings in practice, but they aren't bit-identical.

## D14. On-disk embedding cache (doubles as a shareable snapshot)

- **Decision:** after embedding, vectors are saved to `.cache/embeddings/<model>-<fingerprint>.npz`. The fingerprint hashes every embedded text plus the model name and revision, and the chunk ids are stored and checked on load.
- **Why:** embedding is the only slow step. With the cache:
  - re-indexing is instant;
  - several models can be compared offline (`index --model X --no-upload`) without recomputing;
  - the file can be published (e.g. as a GitHub Release asset) so that people without a GPU skip the step.
- **Correctness:** any change to the source, the chunker or the model changes the fingerprint, so a stale cache can't be used by mistake (covered by tests).
- **Size:** ~1,500 × 1,024 float32 ≈ 6 MB per model.

## D15. Atlas storage: idempotent upsert + vector index created from code

- **Decision:**
  - documents are upserted by stable `_id` (`ReplaceOne(upsert=True)`, in bulk), and stale ids are deleted;
  - the `vectorSearch` index (cosine, plus filter fields `folder`, `kind`, `model_family`, `task`) is created or updated by the `index` command, which waits until it's queryable.
- **Why:** re-running `index` converges to the same state (no duplicates). A reviewer never needs to click through the Atlas UI. The filter fields enable pre-filtered search (e.g. only `task: segment`).
- **Sizing:** ~1,500 docs × (1,024 floats + text) ≈ 15–20 MB, well under the 512 MB free tier.
- **Trade-off:** `replace_all` treats the collection as holding one index version. Multiple versions or models side by side would need separate collections, or a version field in every query.

---

## Known limitations (running list)

- **Atlas database user has the Atlas Admin role** (created by Atlas' automated security setup). Acceptable for the assignment. In production: a least-privilege user with `readWrite` on the app database only, and ideally separate credentials for the indexer (write) and the chat app (read).
- **Retrieval only covers `models/`, `engine/` and `data/`** (assignment scope). Questions about `ultralytics/utils`, `nn` or `cfg` (e.g. default hyperparameters in `default.yaml`, or `EarlyStopping` in `utils/torch_utils.py`) can't be grounded. The assistant must say so rather than guess.
- **Free-tier services** (Atlas M0, OpenRouter `:free` models) have no SLA, shared resources and daily quotas.

## Future work (running list)

- **API-served embeddings as an alternative backend.** OpenRouter serves open-source embedding models, e.g. `qwen/qwen3-embedding-8b`: Apache-2.0, 13× larger than our 0.6B, $0.01 per million tokens. Indexing the whole corpus (~500k tokens) would cost about $0.005. This would remove PyTorch and the 1.2 GB model download from the install, and it would index in ~1–2 minutes on any machine. Trade-offs: the account needs paid credits; the provider can't be pinned to a model revision, so vectors may drift; there's a runtime dependency on a third party (we saw upstream 429s on free LLMs); and the code leaves the machine. Not done because the assignment emphasises free tiers and self-contained reproducibility. A pluggable `Embedder` backend would make it a config switch.
