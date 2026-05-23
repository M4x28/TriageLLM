"""Fetch MIETIC dataset from HuggingFace.

Source: jackf7499/MIETIC (9,629 ESI-labeled triage cases, CC-BY-NC-SA-4.0)
Output: data/raw/mietic/<split>.parquet
"""
from common import fetch_hf_source, setup_logging


def main() -> int:
    setup_logging("triagellm.fetch.mietic")
    return fetch_hf_source(
        source_key="mietic",
        repo="jackf7499/MIETIC",
        out_subdir="mietic",
        license_str="CC-BY-NC-SA-4.0",
    )


if __name__ == "__main__":
    raise SystemExit(main())
