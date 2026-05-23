"""Fetch WHO ETAT (Emergency Triage Assessment and Treatment) manual.

Source: WHO AFRO, pediatric emergency ABCD triage protocol.
Output: data/raw/pdf/who_etat.pdf
"""
from common import fetch_pdf_source, setup_logging


def main() -> int:
    setup_logging("triagellm.fetch.etat")
    return fetch_pdf_source(
        source_key="who_etat",
        url=("https://www.afro.who.int/sites/default/files/2017-06/"
             "participant_manual.pdf"),
        out_filename="who_etat.pdf",
        license_str="WHO-CC-BY-NC-SA-3.0-IGO",
    )


if __name__ == "__main__":
    raise SystemExit(main())
