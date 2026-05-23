"""Fetch SATS (South African Triage Scale) training manual PDF.

Source: EMSSA, 4-color + TEWS scoring reference for LMIC deployment.
Output: data/raw/pdf/sats_manual.pdf
"""
from common import fetch_pdf_source, setup_logging


def main() -> int:
    setup_logging("triagellm.fetch.sats")
    return fetch_pdf_source(
        source_key="sats",
        url=("https://emssa.org.za/wp-content/uploads/2011/04/"
             "SATS-Manual-A5-LR-spreads.pdf"),
        out_filename="sats_manual.pdf",
        license_str="EMSSA",
    )


if __name__ == "__main__":
    raise SystemExit(main())
