"""Conservative text cleaning for extracted PDF content.

Cleaning preserves all meaningful content — eligibility conditions,
legal clauses, Devanagari text, scheme names, numbers, dates and
section headings.  It only removes noise introduced by the PDF
extraction process itself.

It does NOT:
- Translate text.
- Summarize or paraphrase.
- Remove legal conditions or eligibility details.
- Strip Devanagari or any non-ASCII characters.
"""

import re
import unicodedata


def clean_page_text(raw_text: str) -> str:
    """Apply conservative normalization to a single page's raw text.

    Processing steps (in order):
    1. Unicode NFC normalization — composing combining characters.
    2. Normalize line endings to ``\\n``.
    3. Replace non-breaking spaces and other whitespace variants with
       a regular space; keep actual newlines.
    4. Remove carriage returns left over from Windows-style PDF text.
    5. Collapse runs of horizontal whitespace on each line to a single space.
    6. Strip trailing whitespace from every line.
    7. Collapse more than two consecutive blank lines to exactly two,
       preserving paragraph breaks.
    8. Strip leading and trailing whitespace from the entire result.

    Args:
        raw_text: Raw text as extracted by PyMuPDF.

    Returns:
        Cleaned text.  An empty string is returned unchanged.
    """
    if not raw_text:
        return raw_text

    # Step 1 — NFC unicode normalization (composes Devanagari combining chars)
    text = unicodedata.normalize("NFC", raw_text)

    # Step 2 — Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 3 — Replace non-breaking spaces and other Unicode whitespace
    # variants (U+00A0, U+202F, U+2009, etc.) with regular spaces,
    # but preserve actual newlines.
    text = re.sub(r"[^\S\n]", " ", text)

    # Step 4 — Collapse runs of horizontal whitespace on each line
    text = re.sub(r" {2,}", " ", text)

    # Step 5 — Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Step 6 — Collapse more than 2 consecutive blank lines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 7 — Final strip
    return text.strip()


def clean_document_pages(raw_pages: list[str]) -> list[str]:
    """Clean a list of raw page texts extracted from a PDF.

    Args:
        raw_pages: List of raw text strings, one per page.

    Returns:
        List of cleaned text strings, in the same order and length.
    """
    return [clean_page_text(page) for page in raw_pages]
