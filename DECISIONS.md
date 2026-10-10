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

- **Decision:** the model is configurable among 4 pre-selected candidates, each pinned to an exact Hugging Face revision. Qwen3-Embedding-0.6B was the provisional default. **The evaluation chose `BAAI/bge-small-en-v1.5`** (see D18).
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
- **Final choice is made with data** in the retrieval evaluation, on our own question set. **The rule is fixed in advance:** pick the best Hit@5 (written as "Recall@5" in the first draft; it means at least one relevant chunk in the top 5). If a smaller model is within 3 points, pick the smaller one.
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
  - Vectors from GPU fp16 and CPU fp32 aren't bit-identical. **Verified for the chosen model** (bge-small, full corpus computed both ways):
    - per-chunk cosine between the two versions ≥ 0.9995;
    - identical Hit@1, Hit@5 and MRR on the 40 evaluation questions;
    - the rank of the expected answer unchanged for every question;
    - 2/40 top-5 lists differ only in the order of near-tied lower positions.
- **Measured full indexing with the chosen model** (bge-small, from scratch): 395 s on the laptop CPU (i7-10750H, peak RAM ~1 GB) vs 104 s on the GTX 1650 Ti.

## D14. On-disk embedding cache (doubles as a shareable snapshot)

- **Decision:** after embedding, vectors are saved to `embeddings/<model>-<fingerprint>.npz`. The fingerprint hashes every embedded text plus the model name and revision, and the chunk ids are stored and checked on load.
- **Why:** embedding is the only slow step. With the cache:
  - re-indexing is instant;
  - several models can be compared offline (`index --model X --no-upload`) without recomputing;
  - the files are committed to the repo (`embeddings/`, ~11 MB for the four candidates), so people without a GPU skip the step. They're small enough for plain git (no LFS or release download), and the fingerprint means a stale file is never used: if the chunks change, the vectors are recomputed.
- **Correctness:** any change to the source, the chunker or the model changes the fingerprint, so a stale cache can't be used by mistake (covered by tests).
- **Size:** 1,532 chunks, compressed: 1.4 MB (bge-small, 384 dimensions) to 3.7 MB (Qwen3-0.6B, 1,024 dimensions).

## D15. Atlas storage: idempotent upsert + vector index created from code

- **Decision:**
  - documents are upserted by stable `_id` (`ReplaceOne(upsert=True)`, in bulk), and stale ids are deleted;
  - the `vectorSearch` index (cosine, plus filter fields `folder`, `kind`, `model_family`, `task`) is created or updated by the `index` command, which waits until it's queryable.
- **Why:** re-running `index` converges to the same state (no duplicates). A reviewer never needs to click through the Atlas UI. The filter fields enable pre-filtered search (e.g. only `task: segment`).
- **Sizing:** ~1,500 docs × (1,024 floats + text) ≈ 15–20 MB, well under the 512 MB free tier.
- **Trade-off:** `replace_all` treats the collection as holding one index version. Multiple versions or models side by side would need separate collections, or a version field in every query.

## D16. Retrieval: over-fetch, deduplicate by symbol, numCandidates = 20× limit, pre-filters

- **Decision:** `Retriever.search` works in four steps:
  1. embed the query with the index's model;
  2. run `$vectorSearch` with `limit = 3k` and `numCandidates = 20 × limit`;
  3. collapse parts of the same symbol, keeping the best-scoring one;
  4. return k distinct code units.

  Optional pre-filters on `folder`, `task` and `kind` are translated into an Atlas `filter` clause.
- **Why over-fetch and deduplicate:** a long function split into 3 parts could otherwise fill 3 of the 5 slots (observed: `ultralytics.engine.exporter` parts 1 and 2 both in a top-5). Distinct units give the LLM more varied evidence.
- **Why 20×:** `$vectorSearch` is approximate (HNSW). MongoDB's guidance is `numCandidates` ≥ 20 × `limit` for near-exact recall. **Measured:** the top-5 overlaps 99.0% with an exact search, with identical Hit@5 (D17).
- **Fail-fast compatibility check:** each document stores `embedding_model`. If the index was built with a different model than the one used for queries, `search` refuses to run. Vectors from different models live in unrelated spaces, and comparing them returns confident nonsense.
- **Trade-offs:**
  - Pre-filters need the user (or a future router) to know the task.
  - Deduplication hides the other parts of a long function. The generation step can fetch sibling parts if needed.

## D17. Retrieval evaluation methodology

- **Questions:** 46 in `eval/questions.toml`, **each sourced from a real Ultralytics GitHub issue** (linked), paraphrased into a self-contained question:
  - 40 in scope: training 12, data 11, prediction 7, validation 2, export 4, specific models 4;
  - 6 out of scope (answers in `utils/`, `nn/` or `cfg/`), kept for the answer-quality evaluation;
  - 14 of the in-scope questions name a symbol or argument, and 26 are plain language.
- **Labels:** the relevant `path::symbol` pairs were written **by reading the source before running any model**, and committed before the first run (`8ca7a9d` predates the results). The loader fails if a label doesn't match a real chunk.
- **Metrics:**
  - Hit@1 and Hit@5: at least one relevant symbol in the top k;
  - MRR@10;
  - a 95% bootstrap CI on Hit@5;
  - median CPU latency to embed a query.

  Hit@5 is the primary metric because the LLM will see about 5 chunks.
- **Exact search for model comparison:** brute-force cosine over each model's cached vectors. This isolates model quality from the ANN index and needs no re-upload per model. A separate run checks that Atlas' ANN returns the same top-5 for the deployed model (99.0% overlap).
- **Known weaknesses:**
  - **Small n.** One question = 2.5 points, and the CIs of all models overlap.
  - **A single labeller** (no inter-annotator agreement).
  - **Labels may be incomplete.** Another chunk could also answer a question and be counted as a miss.
  - The questions were written with knowledge of the codebase.

  Mitigations for later: more questions mined automatically from issues, and a second labeller.

## D18. Embedding model chosen: `BAAI/bge-small-en-v1.5`

- **Results** (`eval/results/retrieval.md`):

  | Model | Params | Hit@1 | Hit@5 | MRR@10 | Query (CPU) |
  |---|---|---|---|---|---|
  | Qwen3-Embedding-0.6B | 596M | 47.5% | **80.0%** | 0.608 | 764 ms |
  | **bge-small-en-v1.5** | **33M** | **52.5%** | 77.5% | **0.626** | **117 ms** |
  | Granite Embedding R2 | 149M | 47.5% | 77.5% | 0.607 | 188 ms |
  | GTE-ModernBERT | 149M | 45.0% | 72.5% | 0.574 | 173 ms |

- **Rule applied:** Qwen has the best Hit@5, but bge-small is 1 question (2.5 points) behind with 18× fewer parameters, so bge-small wins. It also has the best Hit@1 and MRR.
- **What the data says beyond the rule:**
  - **The embedding model isn't the bottleneck.** 4 questions are missed by *every* model: gradient accumulation, plotting augmented samples, the `list` return type, masks without boxes. Their answers are a few lines inside long methods, or use vocabulary different from the question's ("width and height" vs `xywh`). Better chunking or lexical matching would help more than a bigger model.
  - **The subgroups point in different directions:**
    - questions that name a symbol: bge-small 100% vs Qwen 85.7% (n=14);
    - plain-language questions: Qwen 76.9% vs bge-small 65.4% (n=26, a difference of 3 questions).

    Neither difference is significant at this sample size, but the plain-language gap is worth watching. It's also exactly where lexical (hybrid) search would *not* help, so a larger eval set should decide whether to revisit the model.
- **Practical consequences** (they matter for reviewers and production):
  - query embedding is 6.5× faster on CPU (117 vs 764 ms);
  - the model download is ~130 MB instead of 1.2 GB;
  - **full indexing on a laptop CPU becomes practical** (see D13). With Qwen it was 9+ hours.
- **Caveat:** bge-small reads at most 512 tokens, and **498 of 1,532 chunks (32.5%) are truncated** when embedded. Only the header, signature, docstring and the start of the body are seen. It still matches the larger models, which suggests the contextual header and docstrings carry most of the retrieval signal (D11). The LLM always receives the full, untruncated code. A follow-up experiment: a smaller chunk budget (~2,000 chars) to remove the truncation, re-scored with this same evaluation.

## D19. LLM client: our own fallback chain over OpenRouter's free models (implements D7)

- **Decision:** `OpenRouterClient` calls `/chat/completions` with `httpx` and tries an ordered list of models (`LLM_MODELS`).
  - **Falls back** on 429, 5xx, timeouts, provider errors returned with HTTP 200, and empty completions.
  - **Stops** on 401/402, account errors that no other model can fix.
  - **Reports** which model answered and every failed attempt. The CLI prints "answered by X after Y failed".
- **Streaming:** text is shown as it's generated. Fallback is only allowed *before the first token*: once text is on screen, switching models would splice two different answers, so a mid-answer failure is reported as an error.
- **Retries:** `max_retries` retries the *same* model with exponential backoff on transient errors.
  - Interactive use keeps it at 0, because falling back is faster than waiting.
  - The evaluation uses 4 retries, so that a momentary rate limit isn't scored as a bad answer.
- **Alternative ruled out:** OpenRouter's server-side `models: [...]` fallback. It's simpler, but it doesn't treat empty completions as failures and hides the attempts.
- **Settings:**
  - temperature 0, for the most repeatable output. Free providers are still not fully deterministic: the same question was answered once and declined once.
  - max 2,000 output tokens.
  - 90 s timeout.

## D20. Prompt: scoped, cited, honest about gaps, injection-aware

- **Structure:**
  1. A system prompt stating the release and the indexed folders, plus the rules.
  2. Up to 3 previous turns, for follow-ups.
  3. The user message, holding the numbered `<source id=… symbol=… file=… lines=…>` blocks and then the question.
- **Rules:**
  - answer only from the sources;
  - cite `[n]` per statement, and only sources that support it;
  - **answer the supported part and say what is missing**, and use "I couldn't find this in the indexed code" only when nothing is relevant;
  - never guess names, arguments or defaults;
  - concise, with snippets.
- **Why the partial-answer rule:** the first version said "if the sources don't contain the answer, say so". In the first real test the model then refused a question it could mostly answer, because one helper method was missing from the sources. The rule was rewritten before the evaluation was run.
- **Indirect prompt injection:** retrieved text is *data*. The prompt says so, and any `<source>`/`</sources>` sequence inside indexed code is escaped, so the code can't close the block and inject fake context (covered by a test). The risk is low here, because the corpus is Ultralytics' own code. It becomes real as soon as the index includes third-party repos, issues or user uploads.
- **Prompt v2 tried and reverted:** v2 added stricter rules aimed at observed failures:
  - quote values exactly (a search range is not a default);
  - cite only as `[n]` in every factual paragraph;
  - no "usually"/"probably";
  - use history only to resolve references;
  - no invented values in code examples;
  - answer first.

  In manual tests it removed some invented details. But on `val-vs-predict` it answered "I couldn't find this in the indexed code", although a source showed predict's defaults: more caution meant more false refusals. One manual run can't separate this from free-model variance, so v1 was kept. Comparing the two properly needs the answer evaluation (~20 requests on a 10-question subset).
- **k = 6 chunks:** Hit@k on the retrieval evaluation is 77.5% at 5, 80% at 6 and 80% at 8. Six is the knee, at roughly 6–7k prompt tokens.
- **Follow-ups:** a follow-up like "can it be disabled?" is first rewritten by the LLM into a standalone query ("How can NMS be disabled?") using the last turns, so retrieval keeps the subject; a new topic is returned unchanged, which avoids mixing topics (the risk of simply concatenating the previous question).

## D21. Answer evaluation: key facts + an LLM judge from a different model family

- **Reference:** 104 *key facts* for the 40 in-scope questions (2–3 each). Each fact is atomic, essential to a correct answer, and verifiable in the indexed code, with file:line evidence in `eval/questions.toml`. They were drafted per question from the source, then checked against the code and edited (split compound facts, removed peripheral details). Committed before the evaluation ran.
- **Judge:** `nvidia/nemotron-3-super-120b-a12b:free`.
  - It's from a different family from every candidate (Poolside, Cohere, Google), which avoids self-preference bias.
  - In a known-answer smoke test it was correct and fast (3 s). Nemotron-3-Ultra was also correct but 9× slower and stricter. Inkling is unavailable through the API.
  - It receives the question, the same sources, the key facts and the answer, and returns JSON: facts covered, unsupported claims, refused.
- **Metrics:**
  - key-fact coverage;
  - fully-correct rate;
  - grounded rate (no unsupported claims);
  - citations OK (deterministic);
  - out-of-scope refusal rate;
  - **false refusal rate** (declined although the right code was retrieved);
  - errors;
  - latency.
- **Failure attribution:** every answer records whether the relevant code was among its sources. A gap with the code present is a generation problem; without it, it's a retrieval problem.
- **Fair comparison:** sources are retrieved once per question and reused for every candidate, so the models differ only in generation.
- **Pre-registered rule:**
  1. A model that declines fewer than half of the out-of-scope questions is ineligible, because it invents answers.
  2. The best coverage wins.
  3. Within 5 points of it, the more grounded model wins, then the faster one.
- **Judge validation (done):** a blind review by the author covered 10 answers sampled at random (seed 0), with the judge's verdicts not shown.
  - The judge agreed on **25/25 key facts**: Cohen's κ = 1.0, 95% Clopper–Pearson CI for the agreement rate ≈ 86–100%.
  - The sample included the borderline cases. Examples: a fact conveyed only through code the answer quotes; valid alternative solutions (drawing masks with OpenCV instead of `plot(boxes=False)`; `x2 - x1` instead of `xywh`) that don't state the key fact. Human and judge treated them the same way.
  - Scope: this validates key-fact grading only. Refusal detection has one known error (D23); unsupported-claim detection was not reviewed.
- **Known weaknesses:**
  - The judge is an LLM. It's validated against a human on a small sample (25 facts).
  - The key facts reflect one person's notion of "essential", and they reward the canonical answer: a correct but different solution scores 0, so coverage *underestimates* usefulness.
  - 40 questions is a small sample.
  - Free models are non-deterministic run to run.

## D22. OpenRouter credits for the evaluation, not for users

- Free-model limits: 50 requests/day and 20/minute for accounts with fewer than 10 credits purchased, and **1,000/day** once at least 10 credits have been purchased (OpenRouter docs, checked 2026-10-10). Using `:free` models doesn't spend the credits (`total_usage` stayed at 0).
- **The app and the demo need no credits:** one question = one request.
- **The full answer evaluation does:** ~92 requests per model (answer + judge × 46), so ~280 for three candidates. We bought $10 of credits to raise the daily cap, and committed every answer and verdict (`eval/results/answers.jsonl`), so the results can be inspected without re-running them.
- **Ruled out:** a resumable answer cache to spread the run over several days within the free cap. It adds complexity for a one-off cost of $10.

## D23. LLM chosen: `cohere/north-mini-code:free`, fallback chain Cohere → dots-3 → Laguna

- **Candidates** (from the 15 free OpenRouter models on 2026-10-10; none are Qwen, Llama, DeepSeek or Mistral):
  - Cohere North-Mini-Code (code-specialised);
  - Poolside Laguna-S 2.1 (code-specialised);
  - Google Gemma-4-31B (general).

  Excluded:
  - Nemotron, the judge's family;
  - Inkling, which is API-blocked ("agentic harnesses only");
  - LFM-2.5 2.6B, too small;
  - Laguna-XS, which duplicates Laguna.
- **What happened during the run:**
  - **Gemma-4-31B was rate-limited upstream** (Google AI Studio's shared free pool) for 37 of 46 questions, and for 3 of 3 probes afterwards. Gemma-4-26B, the same pool, was blocked as well.
  - Gemma-31B was replaced by **dots-3-note-preview**, which was responding, and evaluated with `--append` (identical sources and judge).
  - An **availability clause was added to the rule after the run**: models failing more than 50% of requests are reported as *unavailable* rather than ranked. Without it, Gemma would have been labelled "ineligible for not declining out-of-scope questions", which would be false: it simply didn't answer. The clause doesn't change the ranking of the other models.
  - **Transient failures** (rate limits, 11 answers across Laguna, Cohere and dots-3) were re-run once under identical conditions (`--retry-failed`). They count as 0 if they still fail.
- **Results** (`eval/results/answers.md`):

  | Model | Coverage | Fully correct | Grounded | Citations OK | OOS declined | False refusals | Median latency |
  |---|---|---|---|---|---|---|---|
  | **Cohere North-Mini-Code** | **53.8%** | **32.5%** | 77.8% | 52.5% | 3/6 | **0%** | 10.1 s |
  | dots-3-note-preview | 42.9% | 20.0% | **82.2%** | **80.0%** | 3/6 | **0%** | 15.9 s |
  | Laguna-S 2.1 | 36.2% | 12.5% | 80.4% | 65.0% | 4/6 | 18.8% | **8.5 s** |

- **Rule applied:** all three are eligible, and Cohere has the best coverage, 11 points clear of the next (above the 5-point tie margin).
- **Findings worth knowing:**
  - **Laguna declines answerable questions.** It said "I couldn't find this" in 18.8% of the cases where the right code was in its sources (e.g. the resume question). That was the main reason it scored lowest.
  - **Cohere misreads, consistently, when the answer isn't indexed.** Asked for the default learning rate (it lives in `cfg/default.yaml`, not indexed), it twice presented the Tuner's search-space lower bounds as "the defaults". The judge flagged it. For a question about activation functions, it answered with an unrelated SAM component.
  - **Cohere and dots-3 are reasoning models.** Part of `max_tokens` goes to hidden reasoning (37 of 97 tokens in a probe). At the evaluation's 2,000-token budget, about 3 of Cohere's and 6 of dots-3's answers were cut off or empty. One example: Cohere returned an empty completion for `val-vs-predict` on every retry. That's not a rate limit. In the app, the fallback chain covers this case, since an empty completion moves to the next model. **Fixed after the evaluation:** the budget is now 4,000, `finish_reason` is recorded, and truncated answers are flagged. The evaluation was not re-run (~184 requests); with Cohere 11 points ahead, the fix can't change the decision.
  - **The judge makes mistakes too.** For `oos-analytics`, Cohere answered "I couldn't find information about usage analytics… in the indexed source code", but the judge marked it as *not declined*. Cohere's true out-of-scope count is therefore 4/6. This motivated the blind human review of the judge (`eval review` / `eval agreement`).
  - **Citation metric corrected after a code review:** the first parser also counted code like `results[0]` or `[640, 640]` as citations. Recomputed from the stored answers with the code-aware parser (no new requests), "Citations OK" went from 45.0/72.5/60.0% to the values above. Rankings are unchanged.
  - **Known judge bias (not corrected):** the judge sometimes repeats a missing key fact as an "unsupported claim", so the grounded rate is somewhat underestimated (e.g. Cohere ~82% instead of 77.8% by a reviewer's recount). It doesn't affect the decision: the 11-point coverage gap decides before the grounded tie-break. The fix would be grading groundedness in a separate call that doesn't see the key facts.
  - **Cohere sometimes cites as 【4】.** The citation parser now accepts it. No evaluation answer used it, so the metrics are unaffected.
- **Fallback chain order:** by coverage, one model per provider (Cohere → dots-3 → Laguna). An outage at one provider, like Google's today, doesn't take the assistant down.
- **What would improve answers most** (measured gaps, not guesses):
  1. **Retrieval:** in 8 of 40 in-scope questions the relevant code was not among the 6 sources (Hit@6 = 80%), so no model could fully answer them. The reranker (D26) later brought this down to 3.
  2. **Indexing `cfg/`**, so default-value questions have real answers to retrieve.
  3. **A stronger, paid model** for production.
  4. **Prompt work against misreading** (e.g. "a search space is not a default").

  Each would be measured with the same evaluation.

## D24. Hybrid search: implemented, measured, off by default

- **What:** an Atlas Search text index (`chunks_text_index`, created and kept in sync by `code-assistant index`) over `qualified_name` (boost 3), `docstring` (boost 2) and `code`. A custom "code" analyzer (whitespace + wordDelimiterGraph + lowercase) splits identifiers, so "nms" matches `agnostic_nms`, `NMSModel` and `self.args.nms`. Vector and keyword lists are merged with **Reciprocal Rank Fusion** (k = 60), which uses ranks only, so cosine and BM25 scales don't matter. If the text search fails, retrieval falls back to vectors only.
- **Rule fixed before measuring:** keep hybrid only if Hit@6 (what the LLM sees) doesn't drop and Hit@5 or MRR improves.
- **Result** (40 questions, live Atlas, `eval/results/search.md`):
  - Hit@1: 52.5% → 57.5%.
  - Hit@5: 77.5% → 75.0%.
  - **Hit@6: 80.0% → 77.5%.**
  - MRR: 0.626 → 0.664.
- **What the data shows:**
  - Hybrid helps when the question has distinctive terms: `val-vs-predict` moved from rank 4 to 1, and `data-rtsp-reconnect` from 10 to 4.
  - It hurts when the terms are generic: `data-rect` moved from 3 to 7, and `val-test-split` from 5 to outside the top 10.
  - The net change is one question, which is noise at n = 40. **Per the rule it stays off** (`HYBRID_SEARCH=false`).
- **Why keep the code:**
  - The candidate pool of top-20 vector plus top-20 keyword results (~32 code units) contains the correct code for **38/40** questions, against 32/40 in today's top 6. Retrieval *finds* the code; it ranks it poorly.
  - That makes the keyword side the candidate generator for the next experiment: a reranker.

- **Local reranker tried, not adopted:** a cross-encoder (`mixedbread-ai/mxbai-rerank-xsmall-v1`, 71M) reordering the ~32 candidates. On the laptop CPU, loading took 23 s and a single reranked search took over a minute, against a budget of under 2 s. It was stopped before scoring all 40 questions. It remains the natural next step for ranking on a GPU or with a hosted reranker, since the right code is among the candidates for 38/40 questions. A hosted reranker was adopted in D26.

## D25. Optional paid model in the chat (free by default)

- **Decision:** a "Use a paid model" checkbox (off by default) puts `PAID_LLM_MODEL` (`openai/gpt-5-nano`) in front of the free chain, for both the follow-up rewrite and the answer.
- **Why:** free models are the weakest link. They often return empty answers (Cohere) and fill gaps by guessing. The assignment allows "any free OpenRouter model or your preference", so the default stays free and the paid model is an opt-in.
- **Cost:** about $0.0004 per answer, and it uses credits rather than the free quota.
- **Robustness:** HTTP 402 (no credits) is no longer fatal: the paid model fails and the free models answer, so the option never breaks the app for someone without credits.
- **Not measured yet:** running the answer evaluation with the paid model first would cost a few cents and would quantify the gain.

## D26. Hosted reranker, on by default

- **What:** the top 20 vector and top 20 keyword results (~33 distinct code units) are reordered by `nvidia/llama-nemotron-rerank-vl-1b-v2:free` through OpenRouter's `/rerank` endpoint, and the best 6 go to the LLM. Each document sent is `path :: qualified_name` plus the code.
- **Why hosted:** the local cross-encoder took over a minute per search on CPU (D24). OpenRouter serves rerankers with the same API key, so there's no new account and no extra setup step.
- **Candidates and cost:** OpenRouter lists 9 rerankers. Three were measured, from one real call each for the price:
  - NVIDIA nemotron-rerank 1B `:free` ($0);
  - Voyage rerank-2.5-lite ($0.02 per M tokens, a retrieval specialist);
  - Qwen3-Reranker-8B ($0.20 per M tokens, open weights).

  Cohere rerank-4-fast was skipped: it costs $0.002 per search, more than Voyage-lite.
- **Rule fixed before measuring:** adopt only if Hit@6 improves by at least 2 questions and the added latency stays under ~2 s.
- **Result** (40 questions):

  | | Hit@1 | Hit@6 | MRR@10 | Added latency (median) | Cost of the run |
  |---|---|---|---|---|---|
  | Vector only | 21 | 32 | 0.628 | – | – |
  | **NVIDIA `:free`** | **25** | **37** | 0.738 | 0.5 s | $0 |
  | Voyage 2.5-lite | 24 | 35 | 0.713 | 0.4 s | $0.011 |
  | Qwen3-Reranker-8B | 25 | 35 | 0.744 | 1.1 s | $0.13 |

  All three pass the rule. The free one is best on Hit@6 and costs nothing, so it was chosen. The run through the app's code path (`uv run code-assistant eval search` → `eval/results/search.md`) reproduces it: Hit@6 80.0% → **92.5%**, MRR 0.626 → 0.738, 0.68 s median search time.
- **Per question:**
  - It fixes 6: `train-plot-augmented`, `data-multi-rtsp`, `data-rtsp-reconnect`, `predict-box-size`, `predict-plot-no-boxes` and `model-sam2-dynamic`.
  - It loses 1: `val-test-split` (rank 5 → 8).
  - It cannot fix `train-freeze` and `train-grad-accum`, whose code isn't among the candidates.
  - 6 wins against 1 loss is a strong signal but not significant at n = 40 (sign test p ≈ 0.12).
- **Trade-offs:**
  - Free models allow ~20 requests per minute. The first evaluation run got 429 on 19 of 40 questions, so `eval search` now paces the calls and reports any fallback.
  - Each chat question uses one more free request.
  - If the reranker fails, the search falls back to vector order and the question still gets answered. The fallback is logged, not shown in the UI.
  - The reranker's scores are small (about 0.04 for a good match), but only their order matters. The source labels show them with three decimals.
- **Opt-out:** `RERANK_MODEL=` (empty) disables it. The chat's "Rerank the sources" checkbox turns it off per question, e.g. to compare both orders in a demo, and `search --no-rerank` does the same in the CLI.
- **Not re-measured:** the answer evaluation (`eval/results/answers.md`) and `docs/examples.md` were produced with vector-only retrieval.

## D27. Evidence coverage: a stricter retrieval metric

- **Why:** Hit@k counts a question as solved if *one* relevant code unit is retrieved. Answers often need several. `val-vs-predict` is a hit at rank 2 (`Model.val`), yet the answer missed its point because the `conf` defaults in `Model.predict` and `BaseValidator.__init__` weren't retrieved.
- **What:** every question already labels its evidence as `path:start-end` ranges (101 ranges, all inside the indexed folders). **Evidence@6** is the share of those non-blank lines that are in the 6 sources, averaged over questions. **Full evidence@6** is the share of questions where all of it is.
- **How a line counts as retrieved:** it must be in a retrieved function or method chunk with the same path and line range. Class and module summaries report their whole source range but contain only part of it, so for them the line's text must also appear in the chunk (tested; on `val-vs-predict` the `Model` summary spanning lines 35–1275 correctly doesn't cover line 540).
- **Result** (`eval/results/search.md`):

  | | Hit@6 | Evidence@6 | Full evidence@6 |
  |---|---|---|---|
  | Vector only | 80.0% | 50.8% | 7.5% |
  | Hybrid (RRF) | 77.5% | 50.7% | 15.0% |
  | Reranked | 92.5% | **63.3%** | **20.0%** |

- **Reading:**
  - The reranker's gain holds under the stricter metric (+12.5 points).
  - The gap between 92.5% and 20% shows where retrieval falls short: it finds *a* relevant piece, rarely *all* of them.
  - Evidence ranges were labelled generously (some cover a whole method), so Full evidence is a strict lower bound. Evidence@6 is the main number for the next experiments.

---

## Known limitations (running list)

- **Atlas database user has the Atlas Admin role** (created by Atlas' automated security setup). Acceptable for the assignment. In production: a least-privilege user with `readWrite` on the app database only, and ideally separate credentials for the indexer (write) and the chat app (read).
- **Retrieval only covers `models/`, `engine/` and `data/`** (assignment scope). Questions about `ultralytics/utils`, `nn` or `cfg` (e.g. default hyperparameters in `default.yaml`, or `EarlyStopping` in `utils/torch_utils.py`) can't be grounded. The assistant must say so rather than guess.
- **Retrieval misses:** vector search alone misses 9/40 evaluation questions at Hit@5 (77.5%); with the reranker (D26), 3/40 miss the top 6. The main failure modes are:
  - logic buried in long methods (`_do_train`, `_setup_train`);
  - a vocabulary gap between user language and API names;
  - near-duplicate model families (SAM2 vs SAM3 predictors).

  See the failure analysis in `eval/results/retrieval.md`.
- **The free LLM catalogue rotates:** models appear and disappear, and none of today's free models are Qwen or Llama (all of those are paid on OpenRouter). The model list is configuration (`LLM_MODELS`), and the client falls back through it, but a model chosen today can disappear later.
- **Follow-up questions** are rewritten into a standalone query before retrieval (one extra LLM call, only when there is history; the UI shows "Searched for: …"). A bad rewrite is possible; if the call fails, the original message is searched.
- **Free-tier services** (Atlas M0, OpenRouter `:free` models) have no SLA, shared resources and daily quotas.

## Future work (running list)

- **API-served embeddings as an alternative backend.** OpenRouter serves open-source embedding models, e.g. `qwen/qwen3-embedding-8b`: Apache-2.0, 13× larger than our 0.6B, $0.01 per million tokens. Indexing the whole corpus (~500k tokens) would cost about $0.005. This would remove PyTorch and the 1.2 GB model download from the install, and it would index in ~1–2 minutes on any machine. Trade-offs: the account needs paid credits; the provider can't be pinned to a model revision, so vectors may drift; there's a runtime dependency on a third party (we saw upstream 429s on free LLMs); and the code leaves the machine. Not done because the assignment emphasises free tiers and self-contained reproducibility. A pluggable `Embedder` backend would make it a config switch.
- **Better candidates for logic buried in long methods.** The two questions the reranker can't fix (`train-freeze`, `train-grad-accum`) are answered by a few lines inside `_setup_train` / `_do_train`, which aren't among the vector or keyword candidates. Smaller chunks for long methods, or a short summary per method, are the next things to measure.
- **A larger, double-labelled evaluation set** (100+ questions mined from issues and discussions), so that differences of a few points become measurable.
- **Chunk-size experiment for small-context models:** a 2,000-char budget removes bge-small's truncation (32.5% of chunks). Compare against the current 4,000.
