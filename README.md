# 🛡️ Computer Security RAG Assistant

A browser-based educational Retrieval-Augmented Generation (RAG) assistant for computer security questions. It retrieves relevant passages from Wikipedia's Computer Security article and sends the context to a language model through a Python backend.

> **Status:** Educational prototype. It is not a production security-advice system, and answer accuracy has not been independently established.

## Features

- Flask backend; the model API key is not exposed to browser JavaScript.
- Wikipedia knowledge-base loading.
- Overlapping text chunking.
- BM25-style keyword retrieval.
- Refusal/fallback when no chunks match.
- Retrieved source links and excerpts.
- Responsive chat interface and dark/light themes.
- Input length limits, request timeouts, error handling, and basic tests.

## Architecture

1. The browser sends a question to `POST /api/chat`.
2. Flask validates the input.
3. The backend retrieves relevant chunks using BM25-style scoring.
4. If no relevant context is found, it returns an insufficient-context response.
5. Otherwise, the backend calls the configured Fireworks AI model.
6. The UI displays the answer and retrieved sources.

## Requirements

- Python 3.10+
- Internet access
- A Fireworks AI API key and access to the configured model

## Run locally

### 1. Create a virtual environment

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure your API key

Copy `.env.example` to `.env` and replace the placeholder with your own Fireworks AI key.

**Never commit `.env` or real credentials.** The `.gitignore` file excludes `.env`.

### 4. Start the application

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

The app attempts to load the Wikipedia article when you open the app or call the status endpoint. If the provider blocks direct browser API calls in your previous version, this backend architecture avoids making model requests from the browser.

## Run tests

```bash
python -m unittest discover -s tests -v
```

Tests cover basic chunking and retrieval behavior. They do not measure generated-answer accuracy.

## Configuration

| Variable | Purpose |
|---|---|
| `FIREWORKS_API_KEY` | Server-side API credential |
| `FIREWORKS_MODEL` | Model identifier |
| `TOP_K` | Maximum number of chunks supplied to the model |
| `MIN_RETRIEVAL_SCORE` | Minimum ranking score to proceed |
| `MAX_QUESTION_CHARS` | Maximum accepted question length |
| `PORT` | Local development port |

Retrieval scores are ranking scores, not probabilities. The default threshold is a starting value and should be tuned using an evaluation dataset.

## Security notes

- API credentials are read from server-side environment variables.
- The UI uses `textContent` for answer rendering to avoid interpreting model output as HTML.
- Requests have a size limit, question length limit, and outbound timeouts.
- Do not deploy the development server directly to the public internet.
- Before production use, add rate limiting, CSRF protections if cookie-based authentication is introduced, monitoring, stricter outbound/network policies, and a production WSGI deployment.
- Do not treat retrieved documents or model responses as trusted instructions.
- Rotate any key that has ever been committed to a public repository. Removing it from the latest file does not remove it from Git history.

## Current limitations

- The knowledge base is a single Wikipedia article and is fetched at runtime.
- Retrieval is BM25-style lexical ranking, not embedding-based semantic search.
- Ranking scores are not calibrated relevance probabilities.
- The source can change over time.
- Prompt instructions do not guarantee factual correctness or prevent all prompt-injection effects.
- No claim of accuracy is made until a representative evaluation has been run.

## Future work

- Add a curated multi-document knowledge base.
- Compare BM25 with embedding-based retrieval.
- Add a benchmark of questions, expected answers, and supporting passages.
- Measure Recall@k, answer correctness, and faithfulness.
- Add automated integration tests and rate limiting.
- Support source versioning and richer citations.

## Source and provider

- Knowledge source: [Wikipedia — Computer security](https://en.wikipedia.org/wiki/Computer_security)
- Model API provider: [Fireworks AI](https://fireworks.ai/)

## Disclaimer

For learning and demonstration only. Verify security-critical claims against authoritative documentation and trusted sources.
