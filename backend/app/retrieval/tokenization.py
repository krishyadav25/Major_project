"""Deterministic multilingual tokenizer for BM25 search.

Provides robust tokenization across:
- English keywords
- Hindi in Devanagari script (preserving matras and combining characters)
- Hinglish transliterated terms
- Indian currency values (e.g. ₹6,000, Rs. 6000, 6000)
- Numerical eligibility thresholds (e.g. 2.0 hectares, 18-40 years, 100%)
"""

import re
import unicodedata

# Token extraction pattern:
# 1. Devanagari words with combining characters and matras (U+0900 to U+097F)
# 2. Currency symbol: ₹
# 3. Numbers with optional decimals/commas: \d+(?:[.,]\d+)*
# 4. Latin alphanumeric tokens (English & Hinglish): [a-zA-Z0-9]+
_TOKEN_PATTERN = re.compile(r"[\u0900-\u097F]+|₹|\d+(?:[.,]\d+)*|[a-zA-Z0-9]+")


def tokenize_multilingual(text: str) -> list[str]:
    """Tokenize text into search tokens supporting English, Hindi, Hinglish, and numbers.

    Normalizes Unicode via NFC to compose Devanagari diacritics/matras,
    normalizes currency synonyms (Rs., Rs, INR -> ₹), strips thousand-separator
    commas from numbers so exact limits match (e.g. '6,000' -> '6000'), and lowercases
    Latin characters while keeping Devanagari characters unchanged.

    Args:
        text: Raw query or passage string.

    Returns:
        List of normalized string tokens.
    """
    if not text:
        return []

    # 1. Unicode NFC normalization (composes Devanagari combining characters)
    normalized = unicodedata.normalize("NFC", text)

    # 2. Normalize currency synonyms: Rs., Rs, inr -> ₹
    normalized = re.sub(r"\b(?:rs\.|rs|inr)\b", "₹", normalized, flags=re.IGNORECASE)

    # 3. Extract tokens
    raw_tokens = _TOKEN_PATTERN.findall(normalized)

    tokens: list[str] = []
    for token in raw_tokens:
        # Lowercase Latin characters (no-op on Devanagari)
        t = token.lower()

        # If it's a number with commas (e.g. 6,000 or 2,00,000), strip commas
        if re.match(r"^\d+(?:,\d+)+$", t):
            t = t.replace(",", "")

        if t:
            tokens.append(t)

    return tokens
