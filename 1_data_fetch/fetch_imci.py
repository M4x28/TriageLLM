"""Fetch WHO IMCI (Integrated Management of Childhood Illness) chart booklet.

Source: WHO, pediatric decision tree for primary-care triage.
Output: data/raw/pdf/who_imci.pdf
"""
from common import fetch_pdf_source, setup_logging


def main() -> int:
    setup_logging("triagellm.fetch.imci")
    return fetch_pdf_source(
        source_key="who_imci",
        url=("https://cdn.who.int/media/docs/default-source/mca-documents/"
             "child/imci-integrated-management-of-childhood-illness/"
             "imci-in-service-training/imci-chart-booklet.pdf"),
        out_filename="who_imci.pdf",
        license_str="WHO-CC-BY-NC-SA-3.0-IGO",
    )


if __name__ == "__main__":
    raise SystemExit(main())
