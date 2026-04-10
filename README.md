# AI Resumod

Compare your PDF resume against any job posting — get a fit summary, matched requirements, gaps, and concrete edit suggestions powered by an LLM.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Features

- **Upload a PDF resume** — text is extracted and normalized automatically.
- **Paste a job posting URL** — the page is fetched and the job description is extracted using a multi-strategy pipeline (schema.org JSON-LD, trafilatura, raw HTML fallback).
- **LLM-powered analysis** — fit summary, matched requirements, gaps, and actionable resume edit suggestions.
- **Any OpenAI-compatible model** — works with OpenAI, Ollama, LM Studio, vLLM, or any server that speaks the OpenAI chat API.
- **Debug view** — expandable section showing the extracted job text so you can verify what the LLM sees.

## How It Works

```
PDF upload ──► pypdf text extraction ──► normalize ──┐
                                                      ├──► LLM comparison ──► analysis
Job URL ──► httpx fetch ──► extract job text ─────────┘

Extraction cascade:
  1. Parse schema.org JSON-LD (JobPosting)
  2. trafilatura main-content extraction
  3. Raw <body> text fallback
```

## Quick Start

```bash
# install dependencies
uv sync

# configure
cp .env.example .env
# edit .env and add your OPENAI_API_KEY

# run
uv run streamlit run main.py
```

The app opens at `http://localhost:8501`.

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required for OpenAI)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model to use |
| `OPENAI_BASE_URL` | *(unset — uses OpenAI)* | Base URL for an OpenAI-compatible API |

Change `OPENAI_MODEL` to compare results across models (e.g. `gpt-4o`, `gpt-4.1-mini`).

### Using a Local LLM

Most local inference servers expose an OpenAI-compatible API. Set `OPENAI_BASE_URL` and `OPENAI_MODEL`:

```bash
# Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3

# LM Studio
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL=local-model

# vLLM
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=my-model
```

When using a local server, `OPENAI_API_KEY` can be left empty.

## Limitations

- **JavaScript-rendered pages** — sites that load job details entirely via JS (e.g. some SPAs) won't return useful text without a headless browser. The app warns you when this happens.
- **Login-walled postings** — pages behind authentication can't be fetched.
- **Scanned/image PDFs** — only text-based PDFs are supported; OCR is not included.

## License

[MIT](LICENSE)
