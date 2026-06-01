"""
tests/test_content_extractor.py
--------------------------------
Tests for ContentExtractor.  All Drive API, HTTP, Ollama, pdfplumber, and
reverse_geocoder calls are mocked — no credentials or network required.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from drivesort.content_extractor import (
    VISION_FALLBACK_MODEL,
    ContentExtractor,
    _fetch_thumbnail,
)
from drivesort.embedder import Embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(**kwargs) -> MagicMock:
    """Return a mock DriveFile with sensible defaults."""
    f = MagicMock()
    f.id            = kwargs.get("id", "file123")
    f.name          = kwargs.get("name", "test.txt")
    f.modified      = kwargs.get("modified", "2024-01-01T00:00:00Z")
    f.mime_type     = kwargs.get("mime_type", "text/plain")
    f.extension     = kwargs.get("extension", ".txt")
    f.size_bytes    = kwargs.get("size_bytes", 1000)
    f.snippet       = kwargs.get("snippet", "")
    f.description   = kwargs.get("description", "")
    f.thumbnail_link       = kwargs.get("thumbnail_link", "")
    f.image_media_metadata = kwargs.get("image_media_metadata", {})
    f.video_duration_ms    = kwargs.get("video_duration_ms", 0)
    f.text_for_embedding.return_value = kwargs.get(
        "text_for_embedding", f"{f.name} {f.extension}"
    )
    return f


def _make_extractor(tmp_path: Path, **kwargs) -> tuple[ContentExtractor, MagicMock]:
    drive = MagicMock()
    extractor = ContentExtractor(
        drive=drive,
        cache_path=tmp_path / "content_cache.json",
        vision_model=kwargs.get("vision_model", "llava-phi3"),
        pdf_size_limit_bytes=kwargs.get("pdf_size_limit_bytes", 15 * 1024 * 1024),
    )
    return extractor, drive


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_same_file_same_key(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(id="abc", modified="2024-01-01", name="foo.txt")
        assert extractor.cache_key(f) == extractor.cache_key(f)

    def test_different_modified_different_key(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f1 = _make_file(id="abc", modified="2024-01-01", name="foo.txt")
        f2 = _make_file(id="abc", modified="2024-01-02", name="foo.txt")
        assert extractor.cache_key(f1) != extractor.cache_key(f2)

    def test_matches_embedder_cache_key(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(id="abc", modified="2024-01-01", name="foo.txt")
        raw = f"{f.id}|{f.modified}|{f.name}"
        expected = hashlib.sha1(raw.encode()).hexdigest()
        assert extractor.cache_key(f) == expected
        assert extractor.cache_key(f) == Embedder._cache_key(f)


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

class TestExtractCaching:
    def test_cache_hit_skips_extraction(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        f = _make_file(mime_type="application/vnd.google-apps.document")
        key = extractor.cache_key(f)
        extractor._cache[key] = "cached text"

        result = extractor.extract(f)

        assert result == "cached text"
        drive.export_file.assert_not_called()

    def test_cache_miss_stores_result(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        drive.export_file.return_value = b"hello world from doc"
        f = _make_file(mime_type="application/vnd.google-apps.document")

        result = extractor.extract(f)

        key = extractor.cache_key(f)
        assert key in extractor._cache
        assert extractor._cache[key] == result

    def test_cache_persisted_to_disk(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        drive.export_file.return_value = b"content"
        f = _make_file(mime_type="application/vnd.google-apps.document")

        extractor.extract(f)

        cache_file = tmp_path / "content_cache.json"
        assert cache_file.exists()
        stored = json.loads(cache_file.read_text())
        assert extractor.cache_key(f) in stored

    def test_exception_falls_back_to_text_for_embedding(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        drive.export_file.side_effect = RuntimeError("API error")
        f = _make_file(
            mime_type="application/vnd.google-apps.document",
            text_for_embedding="fallback text",
        )

        result = extractor.extract(f)

        assert result == "fallback text"
        assert extractor.cache_key(f) not in extractor._cache


# ---------------------------------------------------------------------------
# Workspace extractors
# ---------------------------------------------------------------------------

class TestWorkspaceExtractors:
    def test_doc_first_500_words(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        words = " ".join(f"word{i}" for i in range(600))
        drive.export_file.return_value = words.encode()
        f = _make_file(name="report.gdoc", mime_type="application/vnd.google-apps.document")

        result = extractor.extract(f)

        drive.export_file.assert_called_once_with(f.id, "text/plain")
        result_words = result.split()
        assert len(result_words) <= 501  # name + 500 content words

    def test_sheet_header_plus_3_rows(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        csv_data = "Name,Age,City\nAlice,30,NYC\nBob,25,LA\nCarol,35,SF\nDave,40,Chicago\n"
        drive.export_file.return_value = csv_data.encode()
        f = _make_file(name="data.gsheet", mime_type="application/vnd.google-apps.spreadsheet")

        result = extractor.extract(f)

        drive.export_file.assert_called_once_with(f.id, "text/csv")
        assert "Name" in result
        assert "Alice" in result
        assert "Dave" not in result  # 5th row excluded

    def test_slide_plain_text(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        drive.export_file.return_value = b"Slide 1 title\nSlide 2 content"
        f = _make_file(name="deck.gslides", mime_type="application/vnd.google-apps.presentation")

        result = extractor.extract(f)

        drive.export_file.assert_called_once_with(f.id, "text/plain")
        assert "Slide 1 title" in result


# ---------------------------------------------------------------------------
# PDF extractor
# ---------------------------------------------------------------------------

class TestPdfExtractor:
    def test_pdf_skipped_when_too_large(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path, pdf_size_limit_bytes=1000)
        f = _make_file(
            mime_type="application/pdf",
            size_bytes=2000,
            text_for_embedding="metadata fallback",
        )

        result = extractor.extract(f)

        assert result == "metadata fallback"
        drive.download_file.assert_not_called()

    def test_pdf_first_page_text(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        drive.download_file.return_value = b"%PDF fake bytes"
        f = _make_file(name="doc.pdf", mime_type="application/pdf", size_bytes=100)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "extracted page text"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor.extract(f)

        assert "extracted page text" in result
        assert "doc.pdf" in result

    def test_pdf_empty_pages_falls_back(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        drive.download_file.return_value = b"%PDF fake bytes"
        f = _make_file(
            name="empty.pdf",
            mime_type="application/pdf",
            size_bytes=100,
            text_for_embedding="metadata fallback",
        )

        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor.extract(f)

        assert result == "metadata fallback"


# ---------------------------------------------------------------------------
# Code extractor
# ---------------------------------------------------------------------------

class TestCodeExtractor:
    def test_code_first_100_lines(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        lines = "\n".join(f"line {i}" for i in range(200))
        drive.download_file.return_value = lines.encode()
        f = _make_file(name="script.py", mime_type="text/x-python", extension=".py")

        result = extractor.extract(f)

        result_lines = result.splitlines()
        assert len(result_lines) <= 101  # name line + 100 code lines
        assert "line 99" in result
        assert "line 100" not in result

    def test_code_extension_included(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        drive.download_file.return_value = b"def foo(): pass"
        f = _make_file(name="main.py", mime_type="text/x-python", extension=".py")

        result = extractor.extract(f)

        assert ".py" in result


# ---------------------------------------------------------------------------
# Image extractor
# ---------------------------------------------------------------------------

class TestImageExtractor:
    def test_image_with_gps_calls_geocoder(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(
            mime_type="image/jpeg",
            thumbnail_link="https://example.com/thumb.jpg",
            image_media_metadata={
                "location": {"latitude": 48.8566, "longitude": 2.3522},
                "cameraMake": "Apple",
                "cameraModel": "iPhone 15",
            },
        )

        with patch("drivesort.content_extractor._fetch_thumbnail", return_value=b"imgbytes"), \
             patch.object(extractor, "_vision_caption", return_value="Eiffel Tower"), \
             patch("reverse_geocoder.search", return_value=[{"name": "Paris", "cc": "FR"}]):
            result = extractor.extract(f)

        assert "Paris" in result
        assert "FR" in result

    def test_image_no_thumbnail_skips_vision(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(
            mime_type="image/jpeg",
            thumbnail_link="",
            image_media_metadata={"cameraMake": "Canon"},
        )

        with patch.object(extractor, "_vision_caption") as mock_caption:
            extractor.extract(f)

        mock_caption.assert_not_called()

    def test_image_metadata_format(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(
            mime_type="image/jpeg",
            thumbnail_link="",
            image_media_metadata={
                "time": "2023:06:15 14:30:00",
                "cameraMake": "Sony",
                "cameraModel": "A7 IV",
                "width": 6000,
                "height": 4000,
            },
        )

        result = extractor.extract(f)

        assert "2023:06:15" in result
        assert "Sony" in result
        assert "6000x4000" in result


# ---------------------------------------------------------------------------
# Video extractor
# ---------------------------------------------------------------------------

class TestVideoExtractor:
    def test_video_caption_and_duration(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(
            name="wedding.mp4",
            mime_type="video/mp4",
            thumbnail_link="https://example.com/thumb.jpg",
            video_duration_ms=3 * 60 * 1000,  # 3 minutes
        )

        with patch("drivesort.content_extractor._fetch_thumbnail", return_value=b"imgbytes"), \
             patch.object(extractor, "_vision_caption", return_value="outdoor ceremony"):
            result = extractor.extract(f)

        assert "wedding.mp4" in result
        assert "outdoor ceremony" in result
        assert "3" in result  # duration hint

    def test_video_no_thumbnail_no_caption(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(
            name="clip.mp4",
            mime_type="video/mp4",
            thumbnail_link="",
            video_duration_ms=30000,
        )

        with patch.object(extractor, "_vision_caption") as mock_caption:
            result = extractor.extract(f)

        mock_caption.assert_not_called()
        assert "clip.mp4" in result


# ---------------------------------------------------------------------------
# Vision captioning
# ---------------------------------------------------------------------------

class TestVisionCaption:
    def test_primary_model_success(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        mock_resp = {"message": {"content": "a sunny beach"}}

        with patch("ollama.chat", return_value=mock_resp):
            result = extractor._vision_caption(b"imgbytes")

        assert result == "a sunny beach"

    def test_primary_fails_fallback_succeeds(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        fallback_resp = {"message": {"content": "a mountain view"}}

        def fake_chat(model, messages):
            if model == "llava-phi3":
                raise ConnectionError("Ollama not running")
            return fallback_resp

        with patch("ollama.chat", side_effect=fake_chat):
            result = extractor._vision_caption(b"imgbytes")

        assert result == "a mountain view"

    def test_both_models_fail_returns_empty(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)

        with patch("ollama.chat", side_effect=ConnectionError("Ollama not running")):
            result = extractor._vision_caption(b"imgbytes")

        assert result == ""


# ---------------------------------------------------------------------------
# GPS geocoding
# ---------------------------------------------------------------------------

class TestGpsGeocoding:
    def test_gps_returns_city_country(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)

        with patch("reverse_geocoder.search", return_value=[{"name": "Auckland", "cc": "NZ"}]):
            result = extractor._gps_to_location(-36.86, 174.76)

        assert "Auckland" in result
        assert "NZ" in result

    def test_gps_coordinate_fallback_on_error(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)

        with patch("reverse_geocoder.search", side_effect=Exception("offline")):
            result = extractor._gps_to_location(-36.86, 174.76)

        assert "-36.86" in result
        assert "174.76" in result

    def test_gps_zero_coordinates_are_valid(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(
            mime_type="image/jpeg",
            thumbnail_link="",
            image_media_metadata={
                "location": {"latitude": 0.0, "longitude": 0.0},
            },
        )
        with patch("reverse_geocoder.search", return_value=[{"name": "Null Island", "cc": "XX"}]):
            result = extractor._format_image_metadata(f)

        assert "Null Island" in result


# ---------------------------------------------------------------------------
# _fetch_thumbnail helper
# ---------------------------------------------------------------------------

class TestFetchThumbnail:
    def test_success_returns_bytes(self):
        mock_resp = MagicMock()
        mock_resp.content = b"jpeg bytes"

        with patch("requests.get", return_value=mock_resp):
            result = _fetch_thumbnail("https://example.com/thumb.jpg")

        assert result == b"jpeg bytes"

    def test_non_200_raises(self):
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                _fetch_thumbnail("https://example.com/missing.jpg")

    def test_fetch_error_propagates_through_extract(self, tmp_path):
        extractor, _ = _make_extractor(tmp_path)
        f = _make_file(
            mime_type="image/jpeg",
            thumbnail_link="https://example.com/thumb.jpg",
            image_media_metadata={},
            text_for_embedding="metadata fallback",
        )

        with patch("drivesort.content_extractor._fetch_thumbnail", side_effect=OSError("network")):
            result = extractor.extract(f)

        assert result == "metadata fallback"


# ---------------------------------------------------------------------------
# Fallback for unknown types
# ---------------------------------------------------------------------------

class TestUnknownType:
    def test_unknown_mime_uses_text_for_embedding(self, tmp_path):
        extractor, drive = _make_extractor(tmp_path)
        f = _make_file(
            mime_type="application/octet-stream",
            extension=".bin",
            text_for_embedding="binary file fallback",
        )

        result = extractor.extract(f)

        assert result == "binary file fallback"
        drive.export_file.assert_not_called()
        drive.download_file.assert_not_called()
