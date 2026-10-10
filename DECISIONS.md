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

---

## Known limitations (running list)

- **Atlas database user has the Atlas Admin role** (created by Atlas' automated security setup). Acceptable for the assignment. In production: a least-privilege user with `readWrite` on the app database only, and ideally separate credentials for the indexer (write) and the chat app (read).
- **Free-tier services** (Atlas M0, OpenRouter `:free` models) have no SLA, shared resources and daily quotas.
