"""
drivesort/content_extractor.py
-------------------------------
Per-type content extraction for richer embeddings.

Sits between DriveClient and Embedder: given a DriveFile, returns an
enriched text string that captures actual content rather than just metadata.

Extraction strategies by type:
  Google Docs/Slides  — Drive export API (text/plain), first 500 words
  Google Sheets       — Drive export API (text/csv), header + 3 rows
  PDFs                — download + pdfplumber first page (≤15 MB only)
  Code files          — download, first 100 lines
  Images              — imageMediaMetadata (GPS→location, EXIF) + vision LLM caption
  Videos              — vision LLM caption of thumbnail + duration hint
  Everything else     — falls back to DriveFile.text_for_embedding()

Results are cached to data/content_cache.json using the same SHA1 key as
the embedding cache, so changing a file invalidates both caches together.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from .drive import DriveClient, DriveFile

CONTENT_CACHE_PATH = Path("data/content_cache.json")
DEFAULT_VISION_MODEL = "llava-phi3"
VISION_FALLBACK_MODEL = "moondream"
CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".java", ".go", ".rb",
    ".rs", ".cpp", ".c", ".cs", ".swift", ".kt",
})
WORKSPACE_DOC   = "application/vnd.google-apps.document"
WORKSPACE_SHEET = "application/vnd.google-apps.spreadsheet"
WORKSPACE_SLIDE = "application/vnd.google-apps.presentation"
VISION_PROMPT = (
    "Describe this image in one sentence for file organisation: "
    "what is shown, where, and what kind of content it is."
)


def _fetch_thumbnail(url: str) -> bytes:
    """Fetch a thumbnail URL and return raw image bytes."""
    import requests
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content


class ContentExtractor:
    """
    Extracts rich text from DriveFile objects for better embeddings.

    Pass an instance to Embedder(..., extractor=...) to enable enriched
    embedding. Caches extracted text to avoid redundant API calls and LLM
    invocations on re-runs.
    """

    def __init__(
        self,
        drive: "DriveClient",
        cache_path: Path = CONTENT_CACHE_PATH,
        vision_model: str = DEFAULT_VISION_MODEL,
        pdf_size_limit_bytes: int = 15 * 1024 * 1024,
    ) -> None:
        self._drive              = drive
        self._cache_path         = cache_path
        self._vision_model       = vision_model
        self._pdf_size_limit     = pdf_size_limit_bytes
        self._cache: dict[str, str] = self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, f: "DriveFile") -> str:
        """
        Return enriched text for f. Uses content cache; falls back to
        f.text_for_embedding() on any error.
        """
        key = self.cache_key(f)
        if key in self._cache:
            return self._cache[key]

        try:
            text = self._extract_uncached(f)
        except Exception as exc:
            print(
                f"[content_extractor] warn: {f.name!r}: {exc}",
                file=sys.stderr,
            )
            return f.text_for_embedding()

        self._cache[key] = text
        self._save_cache()
        return text

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _extract_uncached(self, f: "DriveFile") -> str:
        if f.mime_type == WORKSPACE_DOC:
            return self._extract_workspace_doc(f)
        if f.mime_type == WORKSPACE_SHEET:
            return self._extract_workspace_sheet(f)
        if f.mime_type == WORKSPACE_SLIDE:
            return self._extract_workspace_slide(f)
        if f.mime_type == "application/pdf":
            return self._extract_pdf(f)
        if f.extension in CODE_EXTENSIONS:
            return self._extract_code(f)
        if f.mime_type.startswith("image/"):
            return self._extract_image(f)
        if f.mime_type.startswith("video/"):
            return self._extract_video(f)
        return f.text_for_embedding()

    # ------------------------------------------------------------------
    # Type-specific extractors
    # ------------------------------------------------------------------

    def _extract_workspace_doc(self, f: "DriveFile") -> str:
        raw = self._drive.export_file(f.id, "text/plain")
        words = raw.decode("utf-8", errors="replace").split()[:500]
        return f.name + " " + " ".join(words)

    def _extract_workspace_sheet(self, f: "DriveFile") -> str:
        raw = self._drive.export_file(f.id, "text/csv")
        reader = csv.reader(io.StringIO(raw.decode("utf-8", errors="replace")))
        rows = [row for _, row in zip(range(4), reader)]
        return f.name + " " + " | ".join(", ".join(r) for r in rows)

    def _extract_workspace_slide(self, f: "DriveFile") -> str:
        raw = self._drive.export_file(f.id, "text/plain")
        words = raw.decode("utf-8", errors="replace").split()[:500]
        return f.name + " " + " ".join(words)

    def _extract_pdf(self, f: "DriveFile") -> str:
        if f.size_bytes > self._pdf_size_limit:
            return f.text_for_embedding()
        import pdfplumber
        raw = self._drive.download_file(f.id)
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            if not pdf.pages:
                return f.text_for_embedding()
            page_text = pdf.pages[0].extract_text() or ""
        return f.name + " " + page_text[:2000]

    def _extract_code(self, f: "DriveFile") -> str:
        raw = self._drive.download_file(f.id)
        lines = raw.decode("utf-8", errors="replace").splitlines()[:100]
        return f.name + " " + f.extension + " " + "\n".join(lines)

    def _extract_image(self, f: "DriveFile") -> str:
        parts = [f.name, self._format_image_metadata(f)]
        if f.thumbnail_link:
            caption = self._vision_caption(_fetch_thumbnail(f.thumbnail_link))
            if caption:
                parts.append(caption)
        return " ".join(p for p in parts if p)

    def _extract_video(self, f: "DriveFile") -> str:
        from .drive import _duration_hint
        parts = [f.name]
        if f.video_duration_ms:
            parts.append(_duration_hint(f.video_duration_ms))
        if f.thumbnail_link:
            caption = self._vision_caption(_fetch_thumbnail(f.thumbnail_link))
            if caption:
                parts.append(caption)
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Vision LLM
    # ------------------------------------------------------------------

    def _vision_caption(self, image_bytes: bytes) -> str:
        """Caption image_bytes via Ollama vision model; returns "" on failure."""
        for model in (self._vision_model, VISION_FALLBACK_MODEL):
            try:
                resp = ollama.chat(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": VISION_PROMPT,
                        "images": [image_bytes],
                    }],
                )
                return resp["message"]["content"].strip()
            except Exception:
                continue
        return ""

    # ------------------------------------------------------------------
    # Image metadata helpers
    # ------------------------------------------------------------------

    def _format_image_metadata(self, f: "DriveFile") -> str:
        meta = f.image_media_metadata
        if not meta:
            return ""
        parts = []
        loc = meta.get("location", {})
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is not None and lon is not None:
            parts.append(self._gps_to_location(lat, lon))
        time_str = meta.get("time", "")
        if time_str:
            parts.append(f"taken {time_str}")
        make  = meta.get("cameraMake", "")
        model = meta.get("cameraModel", "")
        if make or model:
            parts.append(f"with {(make + ' ' + model).strip()}")
        w = meta.get("width")
        h = meta.get("height")
        if w and h:
            parts.append(f"{w}x{h}")
        return " ".join(parts)

    def _gps_to_location(self, lat: float, lon: float) -> str:
        try:
            import reverse_geocoder as rg
            results = rg.search([(lat, lon)], verbose=False)
            if results:
                r = results[0]
                return f"in {r['name']}, {r['cc']}"
        except Exception:
            pass
        return f"at {lat:.2f},{lon:.2f}"

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(f: "DriveFile") -> str:
        raw = f"{f.id}|{f.modified}|{f.name}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _load_cache(self) -> dict[str, str]:
        if self._cache_path.exists():
            return json.loads(self._cache_path.read_text())
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))
