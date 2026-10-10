# Ultralytics Code Assistant

A retrieval-augmented code assistant that answers questions about [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) by retrieving relevant source code and generating grounded answers.

> 🚧 Work in progress.

## Prerequisites

| Tool | Version | Why | Install |
|------|---------|-----|---------|
| [uv](https://docs.astral.sh/uv/) | ≥ 0.13.0 | Package and Python version manager (required by the assignment) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

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

## Usage

_TODO_

## Example questions

_TODO: 3–5 questions from real YOLO use cases._

## Design notes

_TODO: chunking strategy, extracted metadata, model choices and rationale, trade-offs._

## Development

```bash
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
```

Design decisions, alternatives considered and known limitations are logged in [DECISIONS.md](DECISIONS.md).

## Future work

_TODO: improvements, scaling to production, critical missing features._
