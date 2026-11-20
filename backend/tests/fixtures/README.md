# Test Fixtures

This directory contains static, minimal test fixtures used by the automated test suite.

## `sample_hindi_scheme.pdf`

- **Purpose:** Validates authentic Unicode Devanagari (Hindi) and mixed Hindi-English text extraction, character counting, and text cleaning across non-Windows operating systems (e.g. Linux CI containers) without relying on host-installed system fonts.
- **Font Used:** [Noto Sans Devanagari](https://github.com/notofonts/devanagari) (Variable TTF, `NotoSansDevanagari[wdth,wght].ttf`) officially maintained by Google and the Noto Project Authors.
- **Font Subsetting:** Font glyphs embedded in `sample_hindi_scheme.pdf` are subsetted to only the exact characters used in the document, resulting in a minimal file size (~12.5 KB).
- **License:** Licensed under the [SIL Open Font License, Version 1.1](FONT_LICENSE_OFL.txt).
  - Copyright 2022 The Noto Project Authors (`https://github.com/notofonts/devanagari`).
  - Under Section 1 of the SIL OFL 1.1, the requirement that font software remain under the OFL does not apply to any document created using the font software, and embedding in documents for redistribution is explicitly permitted.
