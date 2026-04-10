"""AI Resumod – compare a PDF resume against a job posting with an LLM."""

from __future__ import annotations

import html as html_mod
import io
import json
import os
import re
from typing import Any

import httpx
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL")

FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SYSTEM_PROMPT = """\
You are an expert career coach and resume reviewer.
The user will give you:
1. The full text of their resume.
2. The full text of a job posting.

Return a structured analysis in **Markdown** with these sections:
## Fit Summary
A 2-3 sentence overall assessment of how well the resume matches the job.

## Matched Requirements
Bullet list of job requirements the resume clearly satisfies.

## Gaps
Bullet list of job requirements or preferences the resume does NOT cover or only weakly covers.

## Edit Suggestions
Concrete, actionable edits the user should make to their resume. For each suggestion:
- Quote the **original text** from the resume.
- Provide the **rewritten text** that better targets this job.
- If a section or skill is missing entirely, tell the user exactly what to add and where \
(e.g. "Add **Kubernetes** to your Skills section if you have experience with it").
Use the format:  **Before:** "…" → **After:** "…" whenever you are rewording existing text.
"""

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Turn single newlines between non-blank lines into spaces, collapse whitespace."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def pdf_bytes_to_string(data: bytes) -> str:
    """Extract and normalize text from raw PDF bytes."""
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return normalize_text("\n".join(pages))


def _strip_html(value: str) -> str:
    """Strip HTML tags from a string and decode entities."""
    text = BeautifulSoup(value, "lxml").get_text(separator=" ")
    return html_mod.unescape(text).strip()


def _collect_jsonld_job_fields(obj: Any) -> str | None:
    """Recursively search a JSON-LD object for JobPosting data."""
    if isinstance(obj, list):
        for item in obj:
            result = _collect_jsonld_job_fields(item)
            if result:
                return result
        return None

    if not isinstance(obj, dict):
        return None

    obj_type = obj.get("@type", "")
    types = obj_type if isinstance(obj_type, list) else [obj_type]

    if any("JobPosting" in t for t in types):
        fields = [
            "title",
            "description",
            "skills",
            "qualifications",
            "responsibilities",
            "experienceRequirements",
            "educationRequirements",
        ]
        parts: list[str] = []
        for field in fields:
            val = obj.get(field)
            if not val:
                continue
            if isinstance(val, str):
                val = _strip_html(val)
            elif isinstance(val, dict):
                val = _strip_html(json.dumps(val))
            elif isinstance(val, list):
                val = " ".join(
                    _strip_html(v) if isinstance(v, str) else _strip_html(json.dumps(v))
                    for v in val
                )
            parts.append(val)
        return normalize_text("\n\n".join(parts))

    for v in obj.values():
        result = _collect_jsonld_job_fields(v)
        if result:
            return result

    return None


MIN_USEFUL_LENGTH = 120


def extract_job_text_from_html(raw_html: str) -> str:
    """Extract job posting text from HTML using a cascading strategy."""
    soup = BeautifulSoup(raw_html, "lxml")

    # Strategy 1: JSON-LD structured data
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json")}):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        text = _collect_jsonld_job_fields(data)
        if text and len(text) >= MIN_USEFUL_LENGTH:
            return text

    # Strategy 2: trafilatura main-body extraction
    body = trafilatura.extract(raw_html)
    if body and len(body) >= MIN_USEFUL_LENGTH:
        return normalize_text(body)

    # Strategy 3: raw body text
    raw_body = soup.body.get_text(separator=" ") if soup.body else soup.get_text(separator=" ")
    text = normalize_text(raw_body)
    if text:
        return text

    return ""


def fetch_url(url: str) -> str:
    """Fetch a URL and return the response text. Raises on failure."""
    with httpx.Client(
        headers=FETCH_HEADERS,
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def compare_resume_to_job(resume_text: str, job_text: str) -> str:
    """Call the LLM to compare resume vs. job posting."""
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY") or "not-needed",
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Resume\n\n{resume_text}\n\n"
                    f"---\n\n## Job Posting\n\n{job_text}"
                ),
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="AI Resumod", page_icon="📄", layout="centered")
    st.title("📄 AI Resumod")
    st.caption("Compare your resume against a job posting using an LLM")

    with st.sidebar:
        st.header("Settings")
        model_label = OPENAI_MODEL
        if OPENAI_BASE_URL:
            model_label += f" @ {OPENAI_BASE_URL}"
        st.code(model_label, language=None)
        st.caption("Change via `OPENAI_MODEL` / `OPENAI_BASE_URL` in `.env`")

    resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    job_url = st.text_input("Job posting URL", placeholder="https://...")

    if not st.button("Analyze", type="primary"):
        return

    if not resume_file:
        st.error("Please upload a resume PDF first.")
        return
    if not job_url.strip():
        st.error("Please enter a job posting URL.")
        return

    # --- PDF ---
    with st.spinner("Reading PDF…"):
        resume_bytes: bytes = resume_file.getvalue()
        resume_text = pdf_bytes_to_string(resume_bytes)
    if not resume_text.strip():
        st.error("Could not extract any text from the PDF. Is it a scanned image?")
        return

    # --- Fetch job page ---
    with st.spinner("Fetching job posting…"):
        try:
            raw_html = fetch_url(job_url.strip())
        except httpx.HTTPStatusError as exc:
            st.error(f"HTTP {exc.response.status_code} when fetching the URL.")
            return
        except httpx.RequestError as exc:
            st.error(
                f"Could not reach the URL: {exc}. "
                "If this is a login-wall or JS-only page, it may not work "
                "without a headless browser."
            )
            return

    # --- Extract job text ---
    with st.spinner("Extracting job text…"):
        job_text = extract_job_text_from_html(raw_html)
    if not job_text.strip():
        st.error(
            "Could not extract meaningful text from the page. "
            "Sites that require login or render content only with "
            "JavaScript may not work without a headless browser."
        )
        return

    with st.expander("Extracted job text (debug)", expanded=False):
        st.text(job_text[:5000] + ("…" if len(job_text) > 5000 else ""))

    # --- LLM comparison ---
    with st.spinner(f"Analyzing with **{OPENAI_MODEL}**…"):
        try:
            analysis = compare_resume_to_job(resume_text, job_text)
        except Exception as exc:
            st.error(f"LLM call failed: {exc}")
            return

    st.divider()
    st.markdown(analysis)


if __name__ == "__main__":
    main()
