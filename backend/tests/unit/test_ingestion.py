"""Unit tests for the PDF ingestion pipeline.

All test PDFs are generated programmatically using PyMuPDF.
No real government documents are used or committed.

Coverage:
- Successful single/multi-page PDF extraction
- 1-based page number preservation
- Hindi (Devanagari) and English Unicode preservation
- Empty PDF handling (zero pages)
- Text-less (scanned-equivalent) PDF handling
- Invalid path and corrupt file error handling
- Text normalization (cleaning)
- Chunk size, overlap and word boundary behaviour
- Page and document metadata preservation in chunks
- Stable chunk IDs within a document
- Short documents and long single-paragraph pages
- End-to-end ingest_pdf pipeline
"""

import io
from pathlib import Path

import pymupdf
import pytest
from app.ingestion.chunker import ChunkConfig, chunk_document
from app.ingestion.cleaner import clean_page_text
from app.ingestion.extractor import extract_pdf
from app.ingestion.models import (
    DocumentMetadata,
    ExtractedPage,
    ProcessedDocument,
    make_doc_id,
)
from app.ingestion.pipeline import ingest_pdf

# ---------------------------------------------------------------------------
# Helpers — programmatic PDF creation
# ---------------------------------------------------------------------------


def _find_devanagari_font() -> str | None:
    """Locate a Devanagari-capable font on the host system.

    Checks standard Windows and Linux font locations.  Returns None if no
    supported font is available.
    """
    candidates = [
        Path("C:/Windows/Fonts/Nirmala.ttc"),
        Path("C:/Windows/Fonts/NirmalaB.ttc"),
        Path("C:/Windows/Fonts/mangal.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf"),
        Path("/usr/share/fonts/google-noto/NotoSansDevanagari-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _make_pdf_bytes(pages: list[str], fontfile: str | None = None) -> bytes:
    """Create a minimal in-memory PDF with one text block per page.

    Args:
        pages: List of text strings, one per page.  An empty string
               produces a page with no text (simulates a scanned page).
        fontfile: Optional path to a font file (e.g. for Devanagari Unicode).

    Returns:
        Raw PDF bytes.
    """
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            if fontfile:
                page.insert_text((72, 72), text, fontname="deva", fontfile=fontfile, fontsize=11)
            else:
                page.insert_text((72, 72), text, fontsize=11)
    if fontfile:
        doc.subset_fonts()
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


def _write_pdf(tmp_path: Path, name: str, pages: list[str], fontfile: str | None = None) -> Path:
    """Write a programmatically generated PDF to a temp file and return its path."""
    path = tmp_path / name
    path.write_bytes(_make_pdf_bytes(pages, fontfile=fontfile))
    return path


def _make_processed_doc(
    tmp_path: Path,
    pages: list[str],
    name: str = "test.pdf",
    scheme_name: str | None = None,
) -> ProcessedDocument:
    """Extract a generated PDF and attach cleaned pages for use in chunker tests."""
    path = _write_pdf(tmp_path, name, pages)
    doc = extract_pdf(path, scheme_name=scheme_name)
    doc.cleaned_pages = [p.raw_text for p in doc.pages]  # use raw as "cleaned" for simplicity
    return doc


def _make_doc_direct(
    pages_text: list[str],
    name: str = "test.pdf",
    scheme_name: str | None = None,
) -> ProcessedDocument:
    """Build a ProcessedDocument directly — no PDF round-trip required.

    Use this in chunker-only tests where the goal is to test chunking logic,
    not PDF extraction.  This avoids PyMuPDF's text-box size limits, which
    truncate long strings inserted via ``page.insert_text()``.

    Args:
        pages_text: List of cleaned text strings, one per logical page.
        name:       Fake filename used to derive doc_id and metadata.
        scheme_name: Optional government scheme name to attach.

    Returns:
        A fully populated :class:`ProcessedDocument` ready for ``chunk_document``.
    """
    doc_id = make_doc_id(name)
    extracted_pages = [
        ExtractedPage(
            page_number=i + 1,
            raw_text=text,
            char_count=len(text),
            is_empty=len(text.strip()) < 10,
        )
        for i, text in enumerate(pages_text)
    ]
    metadata = DocumentMetadata(
        doc_id=doc_id,
        filename=name,
        filepath=f"/tmp/{name}",
        page_count=len(extracted_pages),
        is_scanned=False,
        scheme_name=scheme_name,
    )
    return ProcessedDocument(
        metadata=metadata,
        pages=extracted_pages,
        cleaned_pages=pages_text,
    )


# ---------------------------------------------------------------------------
# Tests — make_doc_id
# ---------------------------------------------------------------------------


def test_make_doc_id_basic() -> None:
    assert make_doc_id("PM_Kisan_Guidelines.pdf") == "pm_kisan_guidelines"


def test_make_doc_id_with_spaces() -> None:
    assert make_doc_id("scheme guidelines.pdf") == "scheme_guidelines"


def test_make_doc_id_with_hyphens() -> None:
    assert make_doc_id("pm-awas-yojana.pdf") == "pm_awas_yojana"


def test_make_doc_id_sanitizes_punctuation() -> None:
    """Special characters, parentheses, brackets and dots are replaced with underscores."""
    assert make_doc_id("PM_Kisan (2024) [v1.0] @Delhi.pdf") == "pm_kisan_2024_v1_0_delhi"
    assert make_doc_id("Scheme...Guidelines---final!.pdf") == "scheme_guidelines_final"


def test_make_doc_id_collision_resistant_prefix() -> None:
    """Prefixing namespaces identical filenames across different schemes."""
    id1 = make_doc_id("guidelines.pdf", prefix="PM-Kisan")
    id2 = make_doc_id("guidelines.pdf", prefix="Ayushman Bharat")
    assert id1 == "pm_kisan_guidelines"
    assert id2 == "ayushman_bharat_guidelines"
    assert id1 != id2

    # If prefix is already present, it is not redundantly prepended
    id3 = make_doc_id("pm_kisan_guidelines.pdf", prefix="PM-Kisan")
    assert id3 == "pm_kisan_guidelines"


# ---------------------------------------------------------------------------
# Tests — PDF extraction
# ---------------------------------------------------------------------------


def test_extract_single_page(tmp_path: Path) -> None:
    """Successful extraction from a single-page PDF."""
    path = _write_pdf(tmp_path, "single.pdf", ["Hello, PM-Kisan!"])
    doc = extract_pdf(path)

    assert doc.metadata.page_count == 1
    assert len(doc.pages) == 1
    assert "PM-Kisan" in doc.pages[0].raw_text
    assert doc.pages[0].page_number == 1
    assert not doc.pages[0].is_empty
    assert not doc.metadata.is_scanned


def test_extract_multi_page_numbers(tmp_path: Path) -> None:
    """Page numbers must be 1-based and sequential."""
    texts = ["Page one content", "Page two content", "Page three content"]
    path = _write_pdf(tmp_path, "multi.pdf", texts)
    doc = extract_pdf(path)

    assert doc.metadata.page_count == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    for i, text in enumerate(texts):
        assert text in doc.pages[i].raw_text


def test_extract_preserves_scheme_name(tmp_path: Path) -> None:
    path = _write_pdf(tmp_path, "scheme.pdf", ["Some content"])
    doc = extract_pdf(path, scheme_name="Ayushman Bharat")

    assert doc.metadata.scheme_name == "Ayushman Bharat"


def test_extract_hindi_unicode_programmatic(tmp_path: Path) -> None:
    """Devanagari text must survive extraction without corruption when a font is present."""
    font = _find_devanagari_font()
    if font is None:
        pytest.skip("No system Devanagari font found for programmatic test generation.")

    hindi = "प्रधानमंत्री किसान सम्मान निधि योजना"
    path = _write_pdf(tmp_path, "hindi.pdf", [hindi], fontfile=font)
    doc = extract_pdf(path)

    assert doc.metadata.page_count == 1
    assert len(doc.pages) == 1
    assert "किसान" in doc.pages[0].raw_text
    assert "प्रधानमंत्री" in doc.pages[0].raw_text


def test_extract_hindi_from_unicode_fixture() -> None:
    """Verify authentic Devanagari text extraction from redistributable fixture PDF.

    This test uses a pre-generated, font-subsetted PDF fixture with authentic
    Hindi text, ensuring verification passes on any operating system (including
    Linux CI containers) without relying on Windows system fonts.
    """
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_hindi_scheme.pdf"
    assert fixture_path.exists(), f"Fixture PDF not found at {fixture_path}"

    doc = extract_pdf(fixture_path, scheme_name="PM-Kisan")

    assert doc.metadata.page_count == 1
    assert len(doc.pages) == 1
    raw = doc.pages[0].raw_text
    # Verify exact Devanagari content and numerical values
    assert "किसान" in raw
    assert "सम्मान" in raw
    assert "पात्र" in raw
    assert "₹6,000" in raw
    assert "annual" in raw and "benefit" in raw

    # Also verify cleaning normalizes non-breaking spaces and preserves multi-word phrases
    cleaned = clean_page_text(raw)
    assert "annual benefit" in cleaned
    assert "पात्र किसानों को प्रति वर्ष ₹6,000" in cleaned


def test_extract_mixed_hindi_english(tmp_path: Path) -> None:
    """Pages with mixed Hindi and English text are both preserved with true Unicode glyphs."""
    font = _find_devanagari_font()
    if font is None:
        pytest.skip("No system Devanagari font found for programmatic test generation.")

    mixed = "PM Kisan – प्रधानमंत्री किसान सम्मान निधि – eligibility ₹2000"
    path = _write_pdf(tmp_path, "mixed.pdf", [mixed], fontfile=font)
    doc = extract_pdf(path)

    text = doc.pages[0].raw_text
    assert "PM" in text
    assert "Kisan" in text
    assert "किसान" in text
    assert "eligibility" in text
    assert "₹2000" in text


def test_extract_empty_pdf_zero_pages(tmp_path: Path) -> None:
    """A PDF with zero pages must be rejected with a clear ValueError."""
    minimal_zero_page_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"xref\n0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer\n<< /Size 3 /Root 1 0 R >>\n"
        b"startxref\n110\n%%EOF\n"
    )
    path = tmp_path / "empty.pdf"
    path.write_bytes(minimal_zero_page_pdf)

    with pytest.raises(ValueError, match="contains no pages"):
        extract_pdf(path)


def test_extract_pdf_closes_file_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the PyMuPDF document handle is closed even if extraction raises an exception."""
    path = _write_pdf(tmp_path, "error.pdf", ["Page content"])

    opened_docs: list[pymupdf.Document] = []
    original_open = pymupdf.open

    def tracking_open(*args, **kwargs):
        doc = original_open(*args, **kwargs)
        opened_docs.append(doc)
        return doc

    monkeypatch.setattr(pymupdf, "open", tracking_open)

    def failing_get_text(*args, **kwargs):
        raise RuntimeError("Simulated corruption during text read")

    monkeypatch.setattr(pymupdf.Page, "get_text", failing_get_text)

    with pytest.raises(RuntimeError, match="Simulated corruption"):
        extract_pdf(path)

    assert len(opened_docs) == 1
    assert opened_docs[0].is_closed


def test_extract_scanned_pdf_flagged(tmp_path: Path) -> None:
    """A PDF whose pages contain only images (no text) must be flagged as scanned."""
    # Pass empty strings — pages exist but have no text, simulating image-only pages.
    path = _write_pdf(tmp_path, "scanned.pdf", ["", "", ""])
    doc = extract_pdf(path)

    assert doc.metadata.is_scanned is True
    assert all(p.is_empty for p in doc.pages)


def test_extract_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract_pdf("/nonexistent/path/to/file.pdf")


def test_extract_wrong_extension(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("not a pdf")
    with pytest.raises(ValueError, match="Expected a .pdf file"):
        extract_pdf(path)


def test_extract_corrupt_pdf(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"This is not a valid PDF file content!!")
    with pytest.raises(RuntimeError, match="PyMuPDF could not open"):
        extract_pdf(path)


def test_extract_doc_id_derived_from_filename(tmp_path: Path) -> None:
    path = _write_pdf(tmp_path, "PM_Kisan_Guidelines.pdf", ["content"])
    doc = extract_pdf(path)
    assert doc.metadata.doc_id == "pm_kisan_guidelines"


def test_extract_metadata_filepath(tmp_path: Path) -> None:
    path = _write_pdf(tmp_path, "meta.pdf", ["content"])
    doc = extract_pdf(path)
    assert doc.metadata.filename == "meta.pdf"
    assert doc.metadata.filepath == str(path.resolve())


# ---------------------------------------------------------------------------
# Tests — text cleaning
# ---------------------------------------------------------------------------


def test_clean_normalizes_line_endings() -> None:
    result = clean_page_text("line1\r\nline2\rline3")
    assert "\r" not in result
    assert "line1" in result
    assert "line2" in result


def test_clean_collapses_multiple_blank_lines() -> None:
    result = clean_page_text("para1\n\n\n\n\npara2")
    assert "\n\n\n" not in result
    assert "para1" in result
    assert "para2" in result


def test_clean_collapses_horizontal_whitespace() -> None:
    result = clean_page_text("word1    word2")
    assert "  " not in result
    assert "word1 word2" in result


def test_clean_preserves_devanagari() -> None:
    hindi = "पात्रता मानदंड – आयु 18 वर्ष"
    result = clean_page_text(hindi)
    assert "पात्रता" in result
    assert "18" in result


def test_clean_preserves_numbers_and_dates() -> None:
    text = "Amount: ₹6,000 per year. Effective from 01-12-2018."
    result = clean_page_text(text)
    assert "₹6,000" in result
    assert "01-12-2018" in result


def test_clean_preserves_section_headings() -> None:
    text = "1. Eligibility Criteria\n2. Benefits\n3. How to Apply"
    result = clean_page_text(text)
    assert "Eligibility Criteria" in result
    assert "Benefits" in result


def test_clean_empty_string() -> None:
    assert clean_page_text("") == ""


def test_clean_whitespace_only() -> None:
    assert clean_page_text("   \n\n  \t  ") == ""


def test_clean_non_breaking_space(tmp_path: Path) -> None:
    text = "word1\u00a0word2"  # non-breaking space
    result = clean_page_text(text)
    assert "\u00a0" not in result
    assert "word1" in result and "word2" in result


# ---------------------------------------------------------------------------
# Tests — chunking
# ---------------------------------------------------------------------------


def test_chunk_basic_produces_chunks(tmp_path: Path) -> None:
    """200 words at chunk_size=100 must produce more than one chunk.

    We build the ProcessedDocument directly (without a PDF round-trip) to
    avoid PyMuPDF's page text-box size limit, which truncates long strings
    inserted with ``page.insert_text()`` and would prevent the chunk_size
    threshold from being exceeded.
    """
    text = "word " * 200  # 200 words, well over chunk_size=100
    doc = _make_doc_direct([text])
    chunks = chunk_document(doc, ChunkConfig(chunk_size=100, chunk_overlap=10))
    assert len(chunks) > 1


def test_chunk_respects_max_size(tmp_path: Path) -> None:
    text = "word " * 200
    doc = _make_processed_doc(tmp_path, [text])
    config = ChunkConfig(chunk_size=80, chunk_overlap=10)
    chunks = chunk_document(doc, config)
    for chunk in chunks:
        assert chunk.metadata.char_count <= config.chunk_size + 20  # slight word-boundary tolerance


def test_chunk_no_word_splitting(tmp_path: Path) -> None:
    """Words must not be split across chunk boundaries."""
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    doc = _make_processed_doc(tmp_path, [text])
    chunks = chunk_document(doc, ChunkConfig(chunk_size=60, chunk_overlap=10))
    for chunk in chunks:
        for word in chunk.text.split():
            assert word.startswith("word"), f"Unexpected token: {word!r}"


def test_chunk_overlap_shares_content(tmp_path: Path) -> None:
    """Consecutive chunks from the same page should share some words (overlap)."""
    text = "word " * 200
    doc = _make_processed_doc(tmp_path, [text])
    config = ChunkConfig(chunk_size=100, chunk_overlap=20)
    chunks = chunk_document(doc, config)
    if len(chunks) >= 2:
        end_words = set(chunks[0].text.split()[-3:])
        start_words = set(chunks[1].text.split()[:5])
        assert end_words & start_words, "Expected some overlap between consecutive chunks"


def test_chunk_preserves_page_number(tmp_path: Path) -> None:
    """Chunk metadata must carry the correct 1-based page number."""
    doc = _make_processed_doc(tmp_path, ["First page content " * 20, "Second page content " * 20])
    chunks = chunk_document(doc, ChunkConfig(chunk_size=80, chunk_overlap=10))
    page_numbers = {c.metadata.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_stable_ids(tmp_path: Path) -> None:
    """Running chunk_document twice on the same document must yield identical IDs."""
    text = "content word " * 100
    doc1 = _make_processed_doc(tmp_path, [text], name="stable.pdf")
    doc2 = _make_processed_doc(tmp_path, [text], name="stable.pdf")

    ids1 = [c.metadata.chunk_id for c in chunk_document(doc1)]
    ids2 = [c.metadata.chunk_id for c in chunk_document(doc2)]
    assert ids1 == ids2


def test_chunk_id_format(tmp_path: Path) -> None:
    """Chunk IDs must follow the pattern '{doc_id}::chunk::{index:04d}'."""
    doc = _make_processed_doc(tmp_path, ["content word " * 20], name="myfile.pdf")
    chunks = chunk_document(doc)
    for i, chunk in enumerate(chunks):
        expected_id = f"myfile::chunk::{i:04d}"
        assert chunk.metadata.chunk_id == expected_id


def test_chunk_preserves_doc_id_and_filename(tmp_path: Path) -> None:
    doc = _make_processed_doc(tmp_path, ["content " * 50], name="pm_kisan.pdf")
    chunks = chunk_document(doc)
    for chunk in chunks:
        assert chunk.metadata.doc_id == "pm_kisan"
        assert chunk.metadata.filename == "pm_kisan.pdf"


def test_chunk_preserves_scheme_name(tmp_path: Path) -> None:
    doc = _make_processed_doc(tmp_path, ["content " * 50], scheme_name="PM-Kisan")
    chunks = chunk_document(doc)
    for chunk in chunks:
        assert chunk.metadata.scheme_name == "PM-Kisan"


def test_chunk_no_empty_chunks(tmp_path: Path) -> None:
    """No returned chunk should have empty or whitespace-only text."""
    text = "word " * 200
    doc = _make_processed_doc(tmp_path, [text])
    chunks = chunk_document(doc, ChunkConfig(chunk_size=60, chunk_overlap=10))
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_chunk_short_document_one_chunk(tmp_path: Path) -> None:
    """A very short document that fits in one chunk should produce exactly one chunk."""
    doc = _make_processed_doc(tmp_path, ["Short text."])
    chunks = chunk_document(doc, ChunkConfig(chunk_size=512, chunk_overlap=64))
    assert len(chunks) == 1
    assert "Short text." in chunks[0].text


def test_chunk_empty_pages_skipped(tmp_path: Path) -> None:
    """Empty pages should produce no chunks."""
    doc = _make_processed_doc(tmp_path, ["", "", "real content here " * 20])
    chunks = chunk_document(doc)
    for chunk in chunks:
        assert chunk.metadata.page_number == 3


def test_chunk_long_paragraph(tmp_path: Path) -> None:
    """A very long single paragraph should be split into multiple chunks.

    Uses direct ProcessedDocument construction to bypass PyMuPDF's
    page text-box truncation, which would prevent the long string from
    being fully embedded in the test PDF.
    """
    long_text = "eligibility " * 300  # ~3600 chars — well over default 512
    doc = _make_doc_direct([long_text])
    chunks = chunk_document(doc, ChunkConfig(chunk_size=200, chunk_overlap=20))
    assert len(chunks) > 2


def test_chunk_zero_overlap_no_word_duplication() -> None:
    """When chunk_overlap = 0, no words should be duplicated between consecutive chunks."""
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    doc = _make_doc_direct([text])
    config = ChunkConfig(chunk_size=25, chunk_overlap=0)
    chunks = chunk_document(doc, config)

    assert len(chunks) > 1
    # Check that consecutive chunks have 0 overlapping words
    for i in range(len(chunks) - 1):
        words_current = set(chunks[i].text.split())
        words_next = set(chunks[i + 1].text.split())
        intersection = words_current & words_next
        assert not intersection, f"Unexpected word overlap with overlap=0: {intersection}"

    # Also verify that recombined words equal the original words exactly
    all_chunk_words: list[str] = []
    for c in chunks:
        all_chunk_words.extend(c.text.split())
    assert all_chunk_words == text.split()


def test_chunk_preserves_numbered_lists_and_linebreaks() -> None:
    """Chunker must preserve line breaks in numbered eligibility criteria."""
    text = (
        "Eligibility Criteria:\n"
        "1. Beneficiary must be a small farmer.\n"
        "2. Landholding must be up to 2.0 hectares.\n"
        "3. Family income must not exceed 200000 per annum.\n"
        "4. Institutional landholders are excluded."
    )
    doc = _make_doc_direct([text])
    chunks = chunk_document(doc, ChunkConfig(chunk_size=120, chunk_overlap=20))

    assert len(chunks) >= 2
    # Verify that multi-line structure is preserved in chunks
    multi_line_chunks = [c for c in chunks if "\n" in c.text]
    assert len(multi_line_chunks) > 0, "Expected line breaks to be preserved in chunks"
    for c in multi_line_chunks:
        lines = c.text.split("\n")
        assert len(lines) >= 2


def test_chunk_preserves_eligibility_criteria_structure() -> None:
    """Bullet points and numerical thresholds must retain their structured layout."""
    text = (
        "PM-Kisan Scheme Guidelines:\n"
        "- Small and marginal farmers with cultivable land up to 2 hectares\n"
        "- Annual income support of ₹6,000 in three equal installments\n"
        "- Direct Benefit Transfer (DBT) to Aadhaar-linked bank accounts\n"
        "- Exclusions: Income tax payers, doctors, engineers, lawyers"
    )
    doc = _make_doc_direct([text])
    chunks = chunk_document(doc, ChunkConfig(chunk_size=160, chunk_overlap=30))

    assert len(chunks) >= 2
    for chunk in chunks:
        # Verify bullets and currency symbols are preserved
        assert "₹" in chunk.text or "hectares" in chunk.text or "Exclusions" in chunk.text
        if "-" in chunk.text:
            assert not chunk.text.startswith("\n")
            assert not chunk.text.endswith("\n")


def test_chunk_config_validation() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ChunkConfig(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError, match="chunk_overlap must be"):
        ChunkConfig(chunk_size=100, chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


# ---------------------------------------------------------------------------
# Tests — end-to-end ingest_pdf
# ---------------------------------------------------------------------------


def test_ingest_pdf_returns_chunks(tmp_path: Path) -> None:
    """End-to-end: ingest_pdf should return a non-empty list of TextChunks."""
    path = _write_pdf(tmp_path, "e2e.pdf", ["PM-Kisan eligibility details. " * 30])
    chunks = ingest_pdf(path, scheme_name="PM-Kisan")

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata.scheme_name == "PM-Kisan"
        assert chunk.text.strip() != ""


def test_ingest_pdf_scanned_raises(tmp_path: Path) -> None:
    """ingest_pdf must raise RuntimeError for scanned (text-less) PDFs."""
    path = _write_pdf(tmp_path, "scanned.pdf", ["", ""])
    with pytest.raises(RuntimeError, match="scanned PDF"):
        ingest_pdf(path)


def test_ingest_pdf_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ingest_pdf("/no/such/file.pdf")


def test_ingest_pdf_chunk_indices_contiguous(tmp_path: Path) -> None:
    """Chunk indices must be contiguous starting at 0."""
    path = _write_pdf(tmp_path, "contiguous.pdf", ["word " * 200, "word " * 200])
    chunks = ingest_pdf(path, config=ChunkConfig(chunk_size=80, chunk_overlap=10))
    indices = [c.metadata.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_ingest_pdf_zero_pages_raises(tmp_path: Path) -> None:
    """ingest_pdf must raise ValueError for zero-page PDFs."""
    minimal_zero_page_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"xref\n0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer\n<< /Size 3 /Root 1 0 R >>\n"
        b"startxref\n110\n%%EOF\n"
    )
    path = tmp_path / "empty_ingest.pdf"
    path.write_bytes(minimal_zero_page_pdf)

    with pytest.raises(ValueError, match="contains no pages"):
        ingest_pdf(path)
