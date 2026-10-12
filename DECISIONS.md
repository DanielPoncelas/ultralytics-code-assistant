# Design decisions

Why each part is built the way it is: the decision, the alternatives considered, and what was given up. Decisions keep their original numbers (D1–D28, referenced from the README and the code) and are grouped by area. The measurements behind the model choices are told in order in [EVALUATION.md](EVALUATION.md).

**Skim the index; open an entry only when you want the details.**

| | Decision | In one line |
|---|---|---|
| **Indexing** | | |
| [D8](#d8-pinned-release-verified-by-commit) | Index a pinned release, verified by commit SHA | Reproducible results; citations never break |
| [D9](#d9-chunk-along-the-syntax-tree) | Chunk along the syntax tree: functions, methods, class and module summaries | Chunks match how people ask about code |
| [D10](#d10-split-long-units-at-statement-boundaries) | Split long units at statement boundaries (4,000 characters, 3-line overlap) | No statement is cut; boundaries don't depend on the model |
| [D11](#d11-a-contextual-header-in-the-embedded-text) | A contextual header in the embedded text | Places an isolated method in the codebase, at no extra cost |
| [D15](#d15-idempotent-upload-indexes-created-from-code) | Idempotent upload; indexes created from code | Re-running converges; no clicks in the Atlas UI |
| **Embeddings** | | |
| [D12](#d12-four-local-open-source-candidates) | Four local open-source candidates, pinned revisions | Choose with data; native `transformers` models only |
| [D18](#d18-bge-small-chosen) | `BAAI/bge-small-en-v1.5` chosen | Within one question of the best, 18× smaller |
| [D28](#d28-qwen3-embedding-8b-through-openrouter-selectable) | Qwen3-Embedding-8B through OpenRouter, selectable in the chat | The best retrieval measured; needs credit, so not the default |
| [D14](#d14-embeddings-cached-and-shipped-in-the-repo) | Embeddings cached and shipped in the repo | Nobody has to recompute them |
| [D13](#d13-cpu-pytorch-by-default-gpu-detected-automatically) | CPU PyTorch by default, GPU detected automatically | A light install; a GPU only matters for re-embedding |
| **Retrieval** | | |
| [D16](#d16-over-fetch-deduplicate-near-exact-search-pre-filters) | Over-fetch, deduplicate, near-exact vector search, pre-filters | Six distinct code units, as recalled by an exact search |
| [D24](#d24-hybrid-search-tried-and-left-off-local-reranker-too-slow) | Hybrid search tried and left off; a local reranker too slow | Ranking, not recall, was the problem |
| [D26](#d26-hosted-reranker-on-by-default) | Hosted reranker, on by default | 5 more questions with related code, for 0.6 s, free |
| **Answer generation** | | |
| [D6](#d6-plain-httpx-no-llm-sdk) | Plain `httpx`, no LLM SDK | Full control of errors and fallback |
| [D7, D19](#d7-d19-an-own-client-with-ordered-fallback) | An own client with ordered fallback over free models | Free models are often rate-limited or return nothing |
| [D20](#d20-a-scoped-cited-injection-aware-prompt-six-sources) | A scoped, cited, injection-aware prompt; six sources | Grounded answers that admit what is missing |
| [D23](#d23-llm-cohere-north-mini-code-then-dots-3-then-laguna) | Cohere North-Mini-Code, then dots-3, then Laguna | The best key-fact coverage; three providers |
| [D25](#d25-an-optional-paid-model) | An optional paid model | Better answers when wanted, free by default |
| **Evaluation** | | |
| [D17](#d17-questions-from-real-issues-labelled-before-any-run) | 46 questions from real issues, labelled before any run | Choices are measured, not guessed |
| [D21](#d21-answers-graded-by-key-facts-and-a-judge-from-another-family) | Answers graded by key facts and a judge from another model family | The judge is checked against a human |
| [D22](#d22-credits-for-the-evaluation-not-for-users) | OpenRouter credits for the evaluation only | The app itself needs none |
| [D27](#d27-stricter-measures-evidence-and-fact-recall) | Stricter measures: evidence and fact recall | Hit@6 overstated what the LLM gets |
| **Project setup** | | |
| [D1](#d1-python-312) | Python 3.12 | Reliable wheels for the ML stack |
| [D2](#d2-uv-with-a-committed-lockfile) | uv with a committed lockfile and a minimum version | The exact environment on any machine |
| [D3](#d3-src-layout) | `src/` layout | Tests run against the installed package |
| [D4](#d4-typed-settings-with-pydantic-settings) | Typed settings with `pydantic-settings` | Clear errors for missing settings; secrets never logged |
| [D5](#d5-a-typer-cli-with-a-check-command) | A Typer CLI with a `check` command | One entry point; a repeatable pre-flight check |

---

## Indexing

### D8. Pinned release, verified by commit

- **Decision:** shallow-clone tag `v8.4.175` and fail unless `HEAD` is commit `69c95d1`.
- **Alternatives:** `main` (Ultralytics released 5 times in the 10 days before this decision); `pip install ultralytics` (pulls PyTorch and OpenCV just to read `.py` files, and the paths wouldn't match GitHub).
- **Why:** a tag can be moved, a commit can't. Citations link to `/blob/<commit>/…`, so they stay valid.
- **Trade-off:** answers describe `v8.4.175`, not today's code. A new release means changing two settings.

### D9. Chunk along the syntax tree

- **Decision:** with Python's `ast`, one chunk per function or method, plus a summary per class (header, docstring, attributes, method signatures) and per module (docstring, imports, constants, signatures).
- **Alternatives:** fixed-size windows (cut functions in half); whole files (up to 4,000 lines); tree-sitter (only pays off for several languages); methods only (nothing would answer "what does class X do?").
- **Why:** people ask about functions, methods and classes. The summaries answer the "big picture" questions.
- **Trade-off:** signatures appear both in summaries and in method chunks. That's intended: they serve different questions.
- <details><summary>Comments and coverage</summary>

  `ast` drops comments, so comment blocks are attached to the statement that follows them. The first full-corpus audit found 45 meaningful comment lines being lost (e.g. `# NOTE: keep this class for backward compatibility`). Now every non-blank line of the 104 files is in some chunk, which a test enforces. Measured before deciding: 1,011 methods (median 18 lines), 127 functions and 188 classes, with 27 units over ~1.5k tokens (the largest 444 lines).
  </details>

### D10. Split long units at statement boundaries

- **Decision:** pack statements up to 4,000 characters, end each part before a statement starts, and repeat 3 lines in the next part.
- **Why characters, not tokens:** chunk boundaries must be identical for every embedding model, so that comparisons only measure the model. 4,000 characters is about 1,000–1,300 tokens, and ~95% of functions fit in one chunk.
- **Trade-off:** a continuation part loses its enclosing block (e.g. that it runs inside a loop). The header and the overlap reduce this. A test checks that every part parses on its own.

### D11. A contextual header in the embedded text

- **Decision:** embed `# File / Module / Symbol / Signature / Part` followed by the code; store the raw code separately for the LLM.
- **Why:** a method body alone is ambiguous (`self.model`: which model?). The header adds the path, class and signature vocabulary users mention ("trainer", "exporter", "segment"), with no extra model calls.
- **Alternative left for later:** an LLM-written summary per chunk ("contextual retrieval"): better for vague questions, but ~1,500 LLM calls per index build.

### D15. Idempotent upload, indexes created from code

- **Decision:** upsert documents by a stable `_id` (a hash of path, symbol and part) and delete ids that no longer exist. `index` creates or updates the vector and keyword indexes and waits until they are queryable.
- **Why:** re-running `index` converges to the same state, and nobody has to configure Atlas by hand. The filter fields (`folder`, `kind`, `model_family`, `task`) allow pre-filtered search.
- **Trade-off:** one collection holds one index version. Several models side by side need separate collections, which is what D28 does.

## Embeddings

### D12. Four local open-source candidates

- **Decision:** compare four models from Hugging Face, each pinned to an exact revision:

  | Model | Size | Context | Why it's a candidate |
  |---|---|---|---|
  | Qwen3-Embedding-0.6B | 596M | 32k | Strong on code retrieval for its size; takes a task instruction |
  | Granite Embedding R2 | 149M | 8k | ModernBERT, trained with code retrieval data |
  | GTE-ModernBERT | 149M | 8k | Same architecture, strong on the CoIR code benchmark |
  | bge-small-en-v1.5 | 33M | 512 | A small generic baseline: measures what bigger models add |

- **Ruled out:** `jina-code-embeddings` (non-commercial licence), `embeddinggemma` (gated), `Qodo-Embed` (custom licence, slow), and `CodeRankEmbed`. CodeRankEmbed needs `trust_remote_code`, and its code crashes with transformers 5. **Rule adopted:** only models that `transformers` supports natively, since remote code runs arbitrary downloaded Python and breaks when libraries move on.
- **Rule fixed before measuring:** the best Hit@5 wins, unless a smaller model is within 3 points.

### D18. bge-small chosen

- **Result:** Qwen3-0.6B 80.0% Hit@5, **bge-small 77.5%**, Granite 77.5%, GTE-ModernBERT 72.5%. bge-small is one question behind, with 18× fewer parameters, the best Hit@1 and MRR, and 117 ms per query on CPU instead of 764 ms.
- **What else the data showed:** four questions are missed by every model, because their answers are a few lines inside long methods. So the embedding model wasn't the main bottleneck.
- **Trade-off:** bge-small reads 512 tokens, so 32.5% of chunks are truncated when embedded. The LLM still gets the full code. D28 revisits this.

### D28. Qwen3-Embedding-8B through OpenRouter, selectable

- **Decision:** offer `qwen/qwen3-embedding-8b` as a second embedding model in the chat, in its own Atlas collection, with bge-small as the default.
- **Why:** the brief asks for an open-source model, not a free one. Qwen3-8B has Apache-2.0 weights, a 32k context (nothing truncated) and is 240× larger than bge-small. On the 40 questions it puts related code in the top 6 for all of them (bge-small: 32), and fixes all 8 of bge's misses without breaking any (p ≈ 0.008).
- **Why not the default:** every question needs OpenRouter credit (about $0.0000002), so a key without credit couldn't search. bge-small is free and local.
- **How:**
  - The provider is pinned (Nebius, no fallback), so documents and questions get vectors from the same deployment.
  - The keyword index is shared with the main collection, since M0 allows 3 search indexes.
  - The vectors are shipped (13.5 MB).
  - A refused query shows a clear message suggesting the other model.
- **Trade-offs:** about 0.4 s more per question, a dependency on one provider, and a hosted revision that can't be pinned (the shipped index vectors are fixed, but query vectors could drift).

### D14. Embeddings cached and shipped in the repo

- **Decision:** save each model's vectors to `embeddings/<model>-<fingerprint>.npz` and commit them. The fingerprint hashes every embedded text plus the model name and revision.
- **Why:** embedding is the only slow step. With the files, `index` only uploads, and the four-model comparison takes about 2 minutes on CPU instead of hours.
- **Safety:** if the source, the chunker or the model changes, the fingerprint changes and the vectors are recomputed, so a stale file is never used (tested). The files total 25 MB, small enough for plain git.

### D13. CPU PyTorch by default, GPU detected automatically

- **Decision:** the default `cpu` dependency group installs CPU-only PyTorch on Linux and Windows (regular wheels on macOS). `--no-group cpu --group cuda` installs CUDA wheels. The device is detected automatically: CUDA, then Apple MPS, then CPU.
- **Why:** with the shipped vectors, answering only embeds one short question, which takes milliseconds on CPU. A GPU only matters for re-embedding the corpus.
- **Why groups, not extras:** uv can make a group default, but not an extra. With extras, a plain `uv run` silently replaced the CPU build with the multi-GB CUDA build.
- <details><summary>Measurements</summary>

  Full indexing with bge-small: 395 s on a laptop CPU (i7-10750H), 104 s on a GTX 1650 Ti. Qwen3-0.6B took ~27 minutes on that GPU, with an ETA over 9 hours on the CPU. GPU (fp16) and CPU (fp32) vectors of bge-small agree to a cosine ≥ 0.9995, with identical evaluation results.
  </details>

## Retrieval

### D16. Over-fetch, deduplicate, near-exact search, pre-filters

- **Decision:** fetch 3 × k results with `numCandidates` = 20 × limit, keep the best-scoring part of each symbol, and return k distinct code units. Optional pre-filters on `folder`, `task` and `kind`.
- **Why:** without deduplication, the parts of one long function could fill half the slots. The 20× rule is MongoDB's guidance for approximate search; measured, the top 5 matches an exact search 99% of the time.
- **Safety:** each chunk records the model that embedded it. Searching with a different model is refused, because vectors from different models are not comparable.

### D24. Hybrid search tried and left off; local reranker too slow

- **Hybrid:** an Atlas Search keyword index whose analyzer splits identifiers ("nms" matches `agnostic_nms`, `NMSModel`), merged with the vector results by Reciprocal Rank Fusion.
  - **The rule:** keep it only if Hit@6 doesn't drop.
  - **The result:** Hit@6 went from 80.0% to 77.5% (Hit@1 52.5% → 57.5%), so it stays off.
  - **Why the code stayed:** the top 20 of each list together contain the right code for 38 of 40 questions. The problem was the ordering, which makes those lists the candidates for a reranker.
- **A local reranker** (`mxbai-rerank-xsmall-v1`, a cross-encoder) took over a minute per question on CPU, against a budget of 2 s. Stopped before the full run.

### D26. Hosted reranker, on by default

- **Decision:** `nvidia/llama-nemotron-rerank-vl-1b-v2:free`, through OpenRouter's `/rerank` with the same key, reorders the ~33 vector and keyword candidates and keeps the best 6.
- **Rule fixed before measuring:** adopt it only with at least 2 more questions at Hit@6 and under ~2 s.
- **Result:** three rerankers passed. The free one won: related code in the top 6 for 37 questions instead of 32. The whole search takes about 0.6 s more per question, of which the reranker call is about 0.5 s.

  | Reranker | Hit@6 | Reranker call (median) | Cost of the run |
  |---|---|---|---|
  | **NVIDIA nemotron 1B (free)** | **37/40** | 0.5 s | $0 |
  | Voyage rerank-2.5-lite | 35/40 | 0.4 s | $0.011 |
  | Qwen3-Reranker-8B | 35/40 | 1.1 s | $0.13 |

- **Trade-offs:** one more free request per question, and free models allow ~20 requests per minute (the evaluation paces its calls). If the reranker fails, the search falls back to vector order and the question is still answered. A switch in the chat turns it off.

## Answer generation

### D6. Plain `httpx`, no LLM SDK

- **Decision:** call OpenRouter's REST API directly.
- **Alternative:** the `openai` SDK pointed at OpenRouter.
- **Why:** full visibility of requests, errors and retries, which the fallback chain needs, and no doubt about the "no frameworks" constraint.

### D7, D19. An own client with ordered fallback

- **Why it's needed:** on the first day, Gemma returned 429 "rate-limited upstream", Nemotron returned an empty answer (a reasoning model that spent its token budget thinking), and Laguna answered. Free models share upstream quotas across all OpenRouter users.
- **Decision:** try an ordered list of models (`LLM_MODELS`).
  - **Fall back** on 429, 5xx, timeouts, provider errors and empty answers.
  - **Stop** on 401, an account error no other model can fix.
  - **Report** which model answered and which failed.
- **Streaming:** fallback only happens before the first token, so two models' answers are never spliced.
- **Alternative ruled out:** OpenRouter's server-side fallback. It doesn't treat empty answers as failures, and it hides the attempts.
- **Settings:** temperature 0, 4,000 output tokens (reasoning models spend part of them thinking), 30 s timeout in the chat.

### D20. A scoped, cited, injection-aware prompt; six sources

- **Rules:**
  - Answer only from the numbered sources, and cite `[n]` per statement.
  - When only part is covered, answer that part and say what is missing. Say "I couldn't find this in the indexed code" only when nothing relevant was retrieved.
  - Never guess names, arguments or defaults.
- **Why the partial-answer rule:** the first version said "if the sources don't contain the answer, say so". The model then refused a question it could mostly answer because one helper was missing.
- **Prompt injection:** retrieved text is data. Any `<source>` tag inside indexed code is escaped, so code can't close the block and inject fake context (tested).
- **A stricter prompt (v2), tried and reverted:** it cut some invented details, but refused a question it could answer. Without a full evaluation, v1 was kept.
- **Six sources:** Hit@k is 77.5% at 5, 80% at 6 and 80% at 8. Six is the knee, at about 6–7k prompt tokens.
- **Follow-ups:** a follow-up such as "can it be disabled?" is first rewritten into a standalone query from the user's previous questions, so the search keeps the subject.

### D23. LLM: Cohere North-Mini-Code, then dots-3, then Laguna

- **Candidates:** the free OpenRouter models on 2026-10-10. None were Qwen, Llama, DeepSeek or Mistral. Nemotron was excluded as the judge's family.
- **Rule fixed before measuring:** a model must decline at least half the out-of-scope questions; then the best key-fact coverage wins; within 5 points, the more grounded and then the faster one.
- **Result:** Cohere 53.8% of key facts, dots-3 42.9%, Laguna 36.2%. Gemma was rate-limited for 37 of 46 questions and was reported as unavailable, not ranked; dots-3 replaced it.
- **Chain order:** by coverage, one model per provider, so one provider's outage doesn't stop the assistant.
- <details><summary>What the run showed</summary>

  - **Laguna declines answerable questions:** 18.8% of the time when the right code was in its sources.
  - **Cohere misreads questions whose answer isn't indexed.** For example, it presented the tuner's search bounds as the default learning rate.
  - **Reasoning models (Cohere, dots-3) can run out of tokens.** At the evaluation's 2,000-token budget, a few answers were cut off. The budget is now 4,000, and cut-off answers are flagged.
  - **Corrections after review:** "Citations OK" was recomputed with a code-aware parser, which no longer counts `results[0]` as a citation. The judge missed one correct refusal. Neither changes the ranking.
  </details>

### D25. An optional paid model

- **Decision:** a switch (off by default) puts `openai/gpt-5-nano` in front of the free chain, for both the follow-up rewrite and the answer.
- **Why:** the free models are the weakest link. The brief allows "a free model or your preference", so the default stays free.
- **Cost and robustness:** about $0.0004 per answer. Without credit (HTTP 402), the free models simply answer.
- **Not measured:** the answer evaluation wasn't re-run with it.

## Evaluation

### D17. Questions from real issues, labelled before any run

- **Decision:** 46 questions, each from a real GitHub issue.
  - **40 in scope:** training 12, data 11, prediction 7, validation 2, export 4, models 4.
  - **6 out of scope.**
- **Labelling:** the code that answers each question, its key facts and its evidence lines were written by reading the source **before** any model was run, and committed first. The loader fails if a label doesn't match a real chunk.
- **Model comparisons use exact search** over each model's saved vectors. This isolates the model from Atlas' approximate index, and a separate check confirms that Atlas agrees with it 99% of the time.
- **Weak spots:** small (one question = 2.5 points); one labeller; labels may miss other valid code.

### D21. Answers graded by key facts and a judge from another family

- **Decision:** 104 key facts (2–3 per question), each verifiable in the code. A judge (`nvidia/nemotron-3-super-120b-a12b:free`, from a different family than every candidate) checks which facts an answer states, which claims the sources don't support, and whether it declined.
- **Fair comparison:** the sources are retrieved once and reused for every model, so the models differ only in generation.
- **Checked against a human:** in a blind review of 10 random answers (25 facts), the judge agreed 25/25 (Cohen's κ = 1.0).
- **Weak spots:** key facts reward the canonical answer, so a correct alternative solution scores 0. Free models vary between runs.

### D22. Credits for the evaluation, not for users

- Free models allow 50 requests a day, or 1,000 once at least $10 of credit has been purchased. Free models don't spend the credit.
- The app needs no credit. The evaluation does (about 92 requests per LLM), so $10 of credit was bought, and every answer and verdict is committed so it can be inspected without re-running anything.

### D27. Stricter measures: evidence and fact recall

- **Why:** Hit@6 counts a question as solved if *one* relevant code unit is retrieved. On `val-vs-predict` it's a hit, but the lines that answer it (`conf=0.25` and `conf=0.001`) were missing.
- **Evidence@6:** the share of the labelled evidence lines that reach the LLM. Summaries count a line only if it's really in their text, because they report their class's whole range.
- **Facts@6:** the judge reads the 6 sources and checks each key fact. A "yes" counts only if it quotes a line that is verbatim in the sources (whitespace ignored, at least 10 characters).
- **Result with the default setup:** Hit@6 92.5%, Facts@6 62.9%, and every key fact for only 42.5% of the questions.
- **Checked against a human:** 76% agreement on 25 facts. The judge was stricter on 6 and more lenient on none, so these are lower bounds, still fair for comparing setups.

## Project setup

### D1. Python 3.12

- **Why:** PyTorch and sentence-transformers ship reliable wheels for it. uv installs the interpreter, so pinning costs reviewers nothing.

### D2. uv with a committed lockfile

- **Decision:** dependencies in `pyproject.toml`, exact versions in `uv.lock`, and `required-version = ">=0.13.0"`.
- **Why:** `uv sync` reproduces the exact environment, and an old uv fails with a clear message.

### D3. `src/` layout

- **Why:** with a flat layout, Python imports the local folder even when packaging is broken, so tests pass locally and fail elsewhere. `src/` makes tests run against the installed package.

### D4. Typed settings with `pydantic-settings`

- **Why:** a missing setting fails at startup with its name, not deep inside a network call. Credentials are `SecretStr`, kept out of logs and tracebacks (tested).

### D5. A Typer CLI with a `check` command

- **Why:** one documented entry point (`index`, `app`, `ask`, `search`, `eval`). `check` validates the settings and pings Atlas and OpenRouter, so setup can be verified in one command, and it doubles as a pre-flight check before a demo.

---

## Known limitations not covered in the README

- **The Atlas database user has the Atlas Admin role**, created by Atlas' automatic security setup. In production: a least-privilege user per role (a reader for the app, a writer for the indexer).
- **The free LLM catalogue changes.** A model chosen today can disappear. The list is configuration (`LLM_MODELS`), and the client falls back through it.
- **A follow-up can be rewritten badly.** The UI shows the rewritten query ("Searched for: …"), and if the rewrite fails, the original message is searched.
