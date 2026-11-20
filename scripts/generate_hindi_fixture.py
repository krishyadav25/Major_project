"""Regenerates the sample_hindi_scheme.pdf test fixture using Noto Sans Devanagari (SIL OFL).

Downloads the authentic Noto Sans Devanagari variable font from Google Fonts,
inserts authentic Hindi and mixed Hindi-English content, subsets the embedded
font to minimize file footprint, and saves the fixture.
"""

import tempfile
import urllib.request
from pathlib import Path

import pymupdf

FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/"
    "notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "tests"
    / "fixtures"
    / "sample_hindi_scheme.pdf"
)

CONTENT_LINES = [
    "प्रधानमंत्री किसान सम्मान निधि (PM-Kisan)",
    "योजना विवरण: पात्र किसानों को प्रति वर्ष ₹6,000 की वित्तीय सहायता दी जाती है।",
    "पात्रता शर्तें:",
    "1. आवेदक भारतीय नागरिक और किसान होना चाहिए।",
    "2. भूमि का विवरण अद्यतन होना चाहिए।",
    "3. संस्थागत भूमिधारक इस योजना के पात्र नहीं हैं।",
    "English Summary: Small and marginal farmers eligible for ₹6000 annual benefit.",
]


def generate_fixture() -> None:
    print(f"Downloading authentic Noto Sans Devanagari from: {FONT_URL}")
    req = urllib.request.Request(FONT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        font_data = resp.read()
    print(f"Downloaded font size: {len(font_data):,} bytes")

    with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tf:
        tf.write(font_data)
        temp_font_path = tf.name

    try:
        doc = pymupdf.open()
        page = doc.new_page()

        text = "\n".join(CONTENT_LINES)
        page.insert_text(
            (54, 72),
            text,
            fontname="noto-sans-devanagari",
            fontfile=temp_font_path,
            fontsize=11,
        )

        # Subset embedded font to only characters used in the document
        if hasattr(doc, "subset_fonts"):
            doc.subset_fonts()

        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        doc.save(
            str(FIXTURE_PATH),
            garbage=4,
            deflate=True,
            clean=True,
        )
        file_size = FIXTURE_PATH.stat().st_size
        print(f"Successfully generated: {FIXTURE_PATH}")
        print(f"Fixture PDF size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

        # Verify embedded font and extractable text
        verify_doc = pymupdf.open(str(FIXTURE_PATH))
        v_page = verify_doc[0]
        fonts = v_page.get_fonts()
        raw_text = v_page.get_text()

        print("Embedded fonts in generated fixture:")
        for f in fonts:
            print(f"  {f}")
            assert "nirmala" not in f[3].lower(), f"Unexpected Nirmala font found: {f}"
            assert "noto" in f[3].lower(), f"Expected Noto font, found: {f}"

        assert "किसान" in raw_text
        assert "सम्मान" in raw_text
        assert "पात्र" in raw_text
        assert "₹6,000" in raw_text
        assert "6,00,000" not in raw_text
        assert "600000" not in raw_text
        assert "annual" in raw_text and "benefit" in raw_text
        print("All verification checks PASSED!")
    finally:
        Path(temp_font_path).unlink(missing_ok=True)


if __name__ == "__main__":
    generate_fixture()
