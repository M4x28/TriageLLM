"""Shared helpers for TriageLLM Phase 1 fetch scripts.

Provides:
  - logging setup (idempotent)
  - sha256 file hashing
  - Manifest read/write with atomic replace
  - HTTP download with retry + size sanity check
  - HuggingFace dataset download to parquet
  - fetch_pdf_source: full-flow wrapper used by PDF entrypoints
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

USER_AGENT = (
    "TriageLLM-DataCollector/0.1 "
    "(educational research; contact: project-maintainer)"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = RAW_DIR / "pdf"
MIETIC_DIR = RAW_DIR / "mietic"
MANIFEST_PATH = RAW_DIR / "manifest.json"

MIN_PDF_BYTES = 50_000
MIN_PARQUET_BYTES = 1_000_000

HTTP_TIMEOUT = 60
HTTP_RETRIES = 3
HTTP_BACKOFF_BASE = 2.0
HTTP_CHUNK = 8192
HASH_CHUNK = 64 * 1024


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(name: str = "triagellm.fetch") -> logging.Logger:
    """Configure root logger once. Safe to call from any entrypoint."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ManifestEntry:
    source: str
    path: str
    url: str
    sha256: str
    size_bytes: int
    fetched_at: str
    license: str


class Manifest:
    """JSON manifest keyed by `<source>::<relative_path>`.

    Multiple files per source are supported (e.g. MIETIC has multiple splits).
    """

    def __init__(self, path: Path = MANIFEST_PATH) -> None:
        self.path = path

    @staticmethod
    def _key(source: str, rel_path: str) -> str:
        return f"{source}::{rel_path}"

    def load(self) -> dict[str, ManifestEntry]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {k: ManifestEntry(**v) for k, v in data.items()}

    def upsert(self, entry: ManifestEntry) -> None:
        entries = self.load()
        entries[self._key(entry.source, entry.path)] = entry
        self._atomic_write(entries)

    def get(self, source: str, rel_path: str) -> ManifestEntry | None:
        return self.load().get(self._key(source, rel_path))

    def _atomic_write(self, entries: dict[str, ManifestEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: asdict(v) for k, v in entries.items()}
        fd, tmp_str = tempfile.mkstemp(
            prefix=".manifest.", suffix=".json.tmp", dir=str(self.path.parent),
        )
        tmp = Path(tmp_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2,
                          sort_keys=True)
            os.replace(tmp, self.path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


def is_cached(source: str, out_path: Path,
              manifest: Manifest | None = None) -> bool:
    """File present on disk AND manifest hash matches recomputed hash."""
    if not out_path.exists():
        return False
    manifest = manifest or Manifest()
    rel = out_path.relative_to(RAW_DIR).as_posix()
    entry = manifest.get(source, rel)
    if entry is None:
        return False
    return sha256_file(out_path) == entry.sha256


# ─────────────────────────────────────────────────────────────────────────────
# HTTP download
# ─────────────────────────────────────────────────────────────────────────────

def download_http(url: str, out_path: Path,
                  min_size_bytes: int = MIN_PDF_BYTES,
                  user_agent: str = USER_AGENT,
                  timeout: int = HTTP_TIMEOUT,
                  retries: int = HTTP_RETRIES) -> Path:
    """Stream URL to disk with retry + min-size sanity check.

    Raises requests.HTTPError or RuntimeError on terminal failure.
    """
    log = logging.getLogger("triagellm.fetch")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent}

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info("GET %s (attempt %d/%d)", url, attempt, retries)
            with requests.get(url, headers=headers, stream=True,
                              timeout=timeout) as r:
                r.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in r.iter_content(HTTP_CHUNK):
                        if chunk:
                            f.write(chunk)
            size = out_path.stat().st_size
            if size < min_size_bytes:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"download too small ({size} < {min_size_bytes} bytes); "
                    "likely error page"
                )
            log.info("saved %.1f KB -> %s", size / 1024, out_path)
            return out_path
        except Exception as e:
            last_exc = e
            log.warning("attempt %d failed: %s", attempt, e)
            if attempt < retries:
                sleep = HTTP_BACKOFF_BASE ** attempt
                log.info("backoff %.1fs", sleep)
                time.sleep(sleep)

    raise RuntimeError(f"all {retries} attempts failed for {url}") from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace dataset download
# ─────────────────────────────────────────────────────────────────────────────

def download_hf(repo: str, out_dir: Path) -> list[Path]:
    """Load HF dataset and dump each split as parquet. Returns paths written."""
    from datasets import load_dataset

    log = logging.getLogger("triagellm.fetch")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading HF dataset %s", repo)
    ds = load_dataset(repo)

    written: list[Path] = []
    for split_name, split in ds.items():
        out = out_dir / f"{split_name}.parquet"
        split.to_parquet(out)
        log.info("split %s: %d rows -> %s", split_name, len(split), out)
        written.append(out)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# High-level wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_pdf_source(source_key: str, url: str, out_filename: str,
                     license_str: str,
                     min_size_bytes: int = MIN_PDF_BYTES) -> int:
    """Full flow for a single-PDF source. Returns exit code (0 ok, 1 fail)."""
    log = logging.getLogger("triagellm.fetch")
    out_path = PDF_DIR / out_filename
    manifest = Manifest()

    if is_cached(source_key, out_path, manifest):
        log.info("[%s] cached (hash match) -> skip", source_key)
        return 0

    try:
        download_http(url, out_path, min_size_bytes=min_size_bytes)
    except Exception as e:
        log.error("[%s] download failed: %s", source_key, e)
        return 1

    entry = ManifestEntry(
        source=source_key,
        path=out_path.relative_to(RAW_DIR).as_posix(),
        url=url,
        sha256=sha256_file(out_path),
        size_bytes=out_path.stat().st_size,
        fetched_at=_now_iso(),
        license=license_str,
    )
    manifest.upsert(entry)
    log.info("[%s] manifest updated", source_key)
    return 0


def fetch_hf_source(source_key: str, repo: str, out_subdir: str,
                    license_str: str,
                    min_size_bytes: int = MIN_PARQUET_BYTES) -> int:
    """Full flow for a HF dataset source (one entry per split)."""
    log = logging.getLogger("triagellm.fetch")
    out_dir = RAW_DIR / out_subdir
    manifest = Manifest()

    expected_splits_cached = True
    if out_dir.exists():
        for parquet in out_dir.glob("*.parquet"):
            if not is_cached(source_key, parquet, manifest):
                expected_splits_cached = False
                break
        if expected_splits_cached and any(out_dir.glob("*.parquet")):
            log.info("[%s] all splits cached -> skip", source_key)
            return 0

    try:
        paths = download_hf(repo, out_dir)
    except Exception as e:
        log.error("[%s] HF load failed: %s", source_key, e)
        return 1

    hf_url = f"hf://{repo}"
    for p in paths:
        size = p.stat().st_size
        if size < min_size_bytes:
            log.error("[%s] split %s too small (%d bytes)",
                      source_key, p.name, size)
            return 1
        entry = ManifestEntry(
            source=source_key,
            path=p.relative_to(RAW_DIR).as_posix(),
            url=hf_url,
            sha256=sha256_file(p),
            size_bytes=size,
            fetched_at=_now_iso(),
            license=license_str,
        )
        manifest.upsert(entry)
    log.info("[%s] %d split(s) recorded in manifest", source_key, len(paths))
    return 0
