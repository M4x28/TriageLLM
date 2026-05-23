"""Shared helpers for TriageLLM Phase 1 step 2 (data cleaning).

Provides:
  - logging setup
  - path constants
  - regex parsers for vital signs + demographics
  - MIETIC instruction-template classifier
  - length filter
  - MinHash near-duplicate dedup
  - record envelope builder + JSONL writer
  - VLLMBatchClient (lazy import; only required for clean_mietic)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def setup_logging(name: str = "triagellm.clean") -> logging.Logger:
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
# Regex library — compiled at module load
# ─────────────────────────────────────────────────────────────────────────────

_RE_BP = re.compile(
    r"(?:blood pressure|bp)[:\s]*?(\d{2,3})\s*/\s*(\d{2,3})",
    re.IGNORECASE,
)
_RE_BP_BARE = re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b")
_RE_HR = re.compile(
    r"(?:heart rate|pulse|hr)[:\s]+(\d{2,3})",
    re.IGNORECASE,
)
_RE_RR = re.compile(
    r"(?:respiratory rate|resp(?:irations?|iratory)?\s*rate|\brr\b)[:\s]+(\d{1,3})",
    re.IGNORECASE,
)
_RE_SPO2 = re.compile(
    r"(?:spo2|o2\s*sat(?:uration)?|oxygen\s*saturation)[:\s]+(\d{1,3})\s*%?",
    re.IGNORECASE,
)
_RE_TEMP_C = re.compile(
    r"(?:temp(?:erature)?)[:\s]+(\d{2,3}(?:\.\d+)?)\s*°?\s*c",
    re.IGNORECASE,
)
_RE_TEMP_F = re.compile(
    r"(?:temp(?:erature)?)[:\s]+(\d{2,3}(?:\.\d+)?)\s*°?\s*f",
    re.IGNORECASE,
)
_RE_AGE = re.compile(
    r"(\d{1,3})[\s-]?year[\s-]?old",
    re.IGNORECASE,
)
_RE_AGE_MONTH = re.compile(
    r"(\d{1,3})[\s-]?month[\s-]?old",
    re.IGNORECASE,
)
_RE_SEX = re.compile(
    r"\b(male|female|man|woman|boy|girl|gentleman|lady)\b",
    re.IGNORECASE,
)


def _fahrenheit_to_celsius(f: float) -> float:
    return round((f - 32) * 5 / 9, 1)


def parse_vitals(text: str) -> dict[str, Any]:
    """Best-effort regex extraction of vital signs. Missing → None."""
    out: dict[str, Any] = {
        "temperature_c": None, "heart_rate": None,
        "resp_rate": None, "spo2": None,
        "sbp": None, "dbp": None,
    }
    if not text:
        return out

    m = _RE_BP.search(text) or _RE_BP_BARE.search(text)
    if m:
        try:
            sbp, dbp = int(m.group(1)), int(m.group(2))
            if 60 <= sbp <= 260 and 30 <= dbp <= 160 and sbp > dbp:
                out["sbp"], out["dbp"] = sbp, dbp
        except ValueError:
            pass

    if (m := _RE_HR.search(text)):
        v = int(m.group(1))
        if 20 <= v <= 250:
            out["heart_rate"] = v

    if (m := _RE_RR.search(text)):
        v = int(m.group(1))
        if 5 <= v <= 80:
            out["resp_rate"] = v

    if (m := _RE_SPO2.search(text)):
        v = int(m.group(1))
        if 50 <= v <= 100:
            out["spo2"] = v

    if (m := _RE_TEMP_C.search(text)):
        v = float(m.group(1))
        if 30.0 <= v <= 45.0:
            out["temperature_c"] = v
    elif (m := _RE_TEMP_F.search(text)):
        v = float(m.group(1))
        if 86.0 <= v <= 113.0:
            out["temperature_c"] = _fahrenheit_to_celsius(v)

    return out


def parse_demographics(text: str) -> dict[str, Any]:
    """Extract age (years preferred, else months/12) and sex."""
    out: dict[str, Any] = {"age_years": None, "sex": "unknown"}
    if not text:
        return out

    if (m := _RE_AGE.search(text)):
        try:
            v = int(m.group(1))
            if 0 <= v <= 120:
                out["age_years"] = v
        except ValueError:
            pass
    elif (m := _RE_AGE_MONTH.search(text)):
        try:
            v = int(m.group(1))
            if 0 < v < 24:
                out["age_years"] = round(v / 12, 1)
        except ValueError:
            pass

    if (m := _RE_SEX.search(text)):
        token = m.group(1).lower()
        if token in {"male", "man", "boy", "gentleman"}:
            out["sex"] = "M"
        elif token in {"female", "woman", "girl", "lady"}:
            out["sex"] = "F"

    return out


# ─────────────────────────────────────────────────────────────────────────────
# MIETIC instruction template classification
# ─────────────────────────────────────────────────────────────────────────────

TASK_TYPES = (
    "esi1_detection", "esi2_detection", "resource_prediction", "other",
)


def classify_task_type(instruction: str) -> str:
    """Match MIETIC instruction template to a discrete task_type."""
    if not instruction:
        return "other"
    s = instruction.lower()
    if "esi-1" in s or "esi level 1" in s or "immediate life-saving" in s:
        return "esi1_detection"
    if "esi level 2" in s or "esi-2" in s or "high-risk" in s:
        return "esi2_detection"
    if "resources" in s and "ed visit" in s:
        return "resource_prediction"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Filters & dedup
# ─────────────────────────────────────────────────────────────────────────────

def length_filter(text: str, min_chars: int = 50, max_chars: int = 5000) -> bool:
    """True if text length within bounds."""
    if not text:
        return False
    n = len(text)
    return min_chars <= n <= max_chars


class MinHashDedup:
    """Near-duplicate detection via MinHashLSH (datasketch).

    Lazy import: datasketch only required when this class is used.
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = 128) -> None:
        from datasketch import MinHash, MinHashLSH
        self._MinHash = MinHash
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._num_perm = num_perm
        self._counter = 0

    def is_duplicate(self, text: str) -> bool:
        mh = self._signature(text)
        key = f"r{self._counter}"
        if self._lsh.query(mh):
            return True
        self._lsh.insert(key, mh)
        self._counter += 1
        return False

    def _signature(self, text: str):
        mh = self._MinHash(num_perm=self._num_perm)
        tokens = re.findall(r"\w+", text.lower())
        for tok in tokens:
            mh.update(tok.encode("utf-8"))
        return mh


# ─────────────────────────────────────────────────────────────────────────────
# Record envelope
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Provenance:
    file: str
    row_idx: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    raw_text_sha256: str = ""


def build_record(
    record_id: str,
    source: str,
    doc_type: str,
    raw_text: str,
    license: str,
    citation: str,
    provenance: Provenance,
    case: dict | None = None,
    section: dict | None = None,
) -> dict[str, Any]:
    """Build unified-envelope record. case XOR section in {None, dict}."""
    if not provenance.raw_text_sha256:
        provenance.raw_text_sha256 = _sha256_short(raw_text)
    return {
        "id": record_id,
        "source": source,
        "doc_type": doc_type,
        "raw_text": raw_text,
        "case": case,
        "section": section,
        "provenance": {
            "file": provenance.file,
            "row_idx": provenance.row_idx,
            "page_start": provenance.page_start,
            "page_end": provenance.page_end,
            "raw_text_sha256": provenance.raw_text_sha256,
        },
        "license": license,
        "citation": citation,
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# PDF section extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

SECTION_PATTERNS = [
    ("signs_symptoms", r"^(?:\d+\.?\s*)?(signs?\s+and\s+symptoms?|symptoms?|clinical\s+features?|clinical\s+presentation)\b"),
    ("diagnosis",      r"^(?:\d+\.?\s*)?(diagnosis|diagnostic\s+criteria|laboratory\s+(?:tests?|diagnosis))\b"),
    ("classification", r"^(?:\d+\.?\s*)?(classification|staging|severity\s+(?:classification|assessment|grading))\b"),
    ("treatment",      r"^(?:\d+\.?\s*)?(treatment|therapy|management|therapeutic\s+approach|case\s+management)\b"),
    ("prevention",     r"^(?:\d+\.?\s*)?(prevention|prophylaxis|vaccination|immunization)\b"),
    ("triage",         r"^(?:\d+\.?\s*)?(triage|emergency\s+(?:signs?|assessment)|priority\s+signs?|red\s+flags?)\b"),
    ("complications",  r"^(?:\d+\.?\s*)?(complications?|adverse\s+(?:effects?|events?)|warning\s+signs?)\b"),
    ("referral",       r"^(?:\d+\.?\s*)?(when\s+to\s+refer|referral\s+criteria|hospitalization\s+criteria)\b"),
]

COMPILED_SECTIONS = [
    (name, re.compile(pat, re.IGNORECASE | re.MULTILINE))
    for name, pat in SECTION_PATTERNS
]


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Return list of per-page text. Empty pages → empty string.

    Uses PyMuPDF (fitz) — handles CID-encoded fonts in WHO PDFs that
    pdfplumber/pdfminer leave as `(cid:NNN)` artifacts. Falls back to
    pdfplumber if PyMuPDF unavailable.
    """
    try:
        import fitz  # PyMuPDF
        pages: list[str] = []
        doc = fitz.open(str(pdf_path))
        try:
            for p in doc:
                pages.append(p.get_text() or "")
        finally:
            doc.close()
        return pages
    except ImportError:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        return pages


def extract_pdf_outline(pdf_path: Path) -> list[tuple[str, int]]:
    """Return [(title, page_number)] from PDF outline if available."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        out: list[tuple[str, int]] = []

        def _walk(items):
            for it in items:
                if isinstance(it, list):
                    _walk(it)
                else:
                    try:
                        title = it.title
                        page_num = reader.get_destination_page_number(it)
                        out.append((title, page_num))
                    except Exception:
                        continue
        _walk(reader.outline)
        return out
    except Exception:
        return []


_JUNK_STRONG_MARKERS = (
    "table of contents",
    "acknowledgements",
    "list of contributors",
    "list of abbreviations",
    "isbn",
    "library of congress",
    "all rights reserved",
    "world health organization 200",
    "world health organization 201",
    "world health organization 202",
    "department of child and adolescent",
    "for further information, please contact",
    "this publication contains",
    "we would like to extend our thanks",
    "we would like to thank",
    "preface",
    "foreword",
    "training materials",
    "course materials",
    "facilitator",
    "nlm classification",
)

_TOC_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+.+?\s+\d+\s*$", re.MULTILINE)
_STANDALONE_NUM_LINE_RE = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
_TRAILING_NUM_LINE_RE = re.compile(r"^.{1,80}\s+\d{1,3}\s*$", re.MULTILINE)


def is_junk_section(text: str) -> bool:
    """Detect TOC, acknowledgements, copyright/editorial filler sections.

    Used to drop sections matched by clinical-header regex that are actually
    front-matter or back-matter (where words like "triage" or "treatment"
    appear in titles but the body is not clinical content).
    """
    if not text or len(text.strip()) < 50:
        return True
    lower = text.lower()
    for marker in _JUNK_STRONG_MARKERS:
        if marker in lower:
            return True
    # TOC patterns: many lines that are bare numbers OR many lines ending
    # in a 1-3 digit page number (catches multi-column TOC where PyMuPDF
    # extracts numbers on separate lines from headers)
    standalone = len(_STANDALONE_NUM_LINE_RE.findall(text))
    if standalone >= 5:
        return True
    toc_lines = _TOC_LINE_RE.findall(text)
    if len(toc_lines) >= 3:
        return True
    trailing = len(_TRAILING_NUM_LINE_RE.findall(text))
    n_lines = max(text.count("\n"), 1)
    if trailing >= 5 and trailing / n_lines > 0.4:
        return True
    return False


def breadcrumb_for_page(outline: list[tuple[str, int]], page: int) -> list[str]:
    """Approximate breadcrumb from flat outline by last entries with page <= target."""
    if not outline:
        return []
    relevant = [t for t, p in outline if p <= page]
    # de-dup adjacent identical titles while preserving order
    seen: set[str] = set()
    crumbs: list[str] = []
    for t in relevant[-5:]:  # cap depth
        if t not in seen:
            seen.add(t)
            crumbs.append(t)
    return crumbs


def split_into_sections(full_text: str, page_offsets: list[int]) -> list[dict]:
    """Split concatenated text by clinical headers.

    page_offsets[i] = char-offset where page i+1 starts. Used to compute
    page_start/page_end for each section.
    """
    matches: list[tuple[int, str, str]] = []
    for name, regex in COMPILED_SECTIONS:
        for m in regex.finditer(full_text):
            matches.append((m.start(), name, m.group(0).strip()))

    sections: list[dict] = []
    if not matches:
        return sections

    matches.sort(key=lambda x: x[0])
    for i, (start, name, header) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        sections.append({
            "section_type": name,
            "header": header,
            "text": body,
            "char_start": start,
            "char_end": end,
        })
    # Annotate page_start/page_end
    for sec in sections:
        sec["page_start"] = _char_to_page(sec["char_start"], page_offsets) + 1
        sec["page_end"] = _char_to_page(sec["char_end"] - 1, page_offsets) + 1
    return sections


def _char_to_page(char_pos: int, page_offsets: list[int]) -> int:
    """Binary search the page index for a char offset (0-based)."""
    lo, hi = 0, len(page_offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if page_offsets[mid] <= char_pos:
            lo = mid
        else:
            hi = mid - 1
    return lo


def sliding_window(text: str, window: int = 1500, overlap: int = 200) -> list[str]:
    """Split oversized section into overlapping chunks."""
    if len(text) <= window:
        return [text]
    chunks: list[str] = []
    step = window - overlap
    for start in range(0, len(text), step):
        end = min(start + window, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# vLLM batch client (lazy import)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VLLMConfig:
    model_name: str = "Qwen/Qwen3.6-27B"
    fallback_model: str = "Qwen/Qwen3.5-9B-Instruct"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85
    dtype: str = "bfloat16"
    enable_thinking: bool = False


class VLLMBatchClient:
    """Wrapper around vLLM offline batch inference with guided JSON output.

    Lazy import of `vllm` so the module is importable in environments
    without CUDA (e.g. local Windows dev box).
    """

    def __init__(self, config: VLLMConfig | None = None) -> None:
        from vllm import LLM, SamplingParams  # type: ignore
        try:
            from vllm.sampling_params import GuidedDecodingParams  # type: ignore
        except ImportError:
            from vllm import GuidedDecodingParams  # type: ignore

        self.cfg = config or VLLMConfig()
        self._SamplingParams = SamplingParams
        self._GuidedDecodingParams = GuidedDecodingParams
        self._log = logging.getLogger("triagellm.clean.vllm")

        self._log.info("loading vLLM model %s", self.cfg.model_name)
        self._llm = LLM(
            model=self.cfg.model_name,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            max_model_len=self.cfg.max_model_len,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            dtype=self.cfg.dtype,
            trust_remote_code=True,
        )

    def batch_extract(
        self,
        prompts: list[str],
        json_schema: dict,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> list[dict]:
        """Run guided-JSON decoding on a batch of prompts."""
        guided = self._GuidedDecodingParams(json=json_schema)
        sp = self._SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            guided_decoding=guided,
        )

        chat_kwargs = (
            {"enable_thinking": False}
            if not self.cfg.enable_thinking and "Qwen3" in self.cfg.model_name
            else {}
        )

        outputs = self._llm.chat(
            [
                [{"role": "user", "content": p}]
                for p in prompts
            ],
            sampling_params=sp,
            chat_template_kwargs=chat_kwargs,
        )

        results: list[dict] = []
        for out in outputs:
            text = out.outputs[0].text.strip()
            try:
                results.append(json.loads(text))
            except json.JSONDecodeError:
                self._log.warning("malformed JSON from model: %s", text[:200])
                results.append({})
        return results
