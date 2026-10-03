"""Generate the PDF documents in sample_data/ from the text defined below.

The PDFs are committed so the sample corpus works out of the box; this script
documents exactly what they contain and lets them be regenerated:

    python scripts/generate_sample_pdfs.py

Requires the ``dev`` extra (fpdf2).
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parents[1] / "sample_data"

# filename -> (title, [pages], each page a list of paragraphs)
DOCUMENTS: dict[str, tuple[str, list[list[str]]]] = {
    "quillon_quarterly_report_q3_2025.pdf": (
        "Quillon Labs Quarterly Report - Q3 2025",
        [
            [
                "Quillon Labs Quarterly Report - Q3 2025",
                "Revenue for the quarter was 4.2 million dollars, up 31 percent year over year. Net revenue "
                "retention reached 118 percent, driven by expansion at existing search customers.",
                "Project Halcyon reached general availability on 6 October 2025. By the end of September, 27 "
                "enterprise tenants had migrated from Project Atlas to Halcyon, and the Atlas Elasticsearch "
                "cluster was shut down.",
                "In August 2025 the company closed a 30 million dollar Series B led by Meridian Capital. The "
                "round funds the European expansion and the next generation of Halcyon relevance work.",
            ],
            [
                "Headcount and operations",
                "Headcount was 64 at quarter end: 51 employees in the Toronto office and 13 remote employees. "
                "Engineering represents 58 percent of headcount.",
                "Risks: the three largest tenants account for 38 percent of revenue. GPU spend exceeded plan "
                "because of embedding backfills for migrating tenants.",
                "Priorities for Q4 2025: harden tenant isolation, research a cross-encoder reranker, and "
                "sign the first two European customers.",
            ],
        ],
    ),
    "brightwater_compliance_audit_2026.pdf": (
        "Brightwater Health HIPAA Compliance Audit Summary",
        [
            [
                "Brightwater Health - HIPAA compliance audit summary, March 2026",
                "Scope: Project Lodestar and the enterprise master patient index. The audit reviewed access "
                "controls, encryption, retention, and the manual review workflow for proposed record merges.",
                "Lodestar satisfied the access control and encryption requirements. The Parquet snapshot "
                "retention finding from the February 2026 security review was remediated on 20 February 2026.",
                "New observation: manual review decisions did not require a second reviewer for merges involving "
                "minors. Brightwater has since introduced dual review for every proposed merge involving a "
                "patient under 18.",
            ],
            [
                "Outcome metrics",
                "After six months of operation, the duplicate patient record rate fell from 8.1 percent to "
                "2.3 percent. The false merge rate measured in the Q1 2026 audit sample was 0.04 percent.",
                "The auditors recommend extending the same record linkage approach to the provider directory "
                "in 2027.",
            ],
        ],
    ),
    "tessellate_wren_whitepaper.pdf": (
        "Offline Language Models for Field Service",
        [
            [
                "Offline language models for field service - Tessellate AI white paper, 2026",
                "Field technicians often work in basements, plant rooms and lift shafts with no connectivity. "
                "Project Wren brings a question-answering assistant to their tablets that runs entirely on the "
                "device.",
                "We compared int8 and int4 weight quantization. The int4 model halved memory use again, but its "
                "error rate on troubleshooting answers rose from 6.1 percent to 11.7 percent, so Wren ships "
                "with int8 weights.",
            ],
            [
                "Field trial",
                "A field trial with 120 technicians at a German elevator maintenance company ran for eight "
                "weeks. The average time to find the right repair procedure dropped from 9 minutes to 3.5 "
                "minutes, and technicians used the assistant on 71 percent of their jobs.",
                "Technicians asked most often for wiring diagrams and error code explanations. Photo-based "
                "search is the most requested feature for the next release.",
            ],
        ],
    ),
    "kestrel_ops_review_q1_2026.pdf": (
        "Kestrel Logistics Operations Review Q1 2026",
        [
            [
                "Kestrel Logistics - operations review, Q1 2026",
                "Kestrel Logistics is headquartered in Rotterdam and operates eleven depots across the "
                "Netherlands, Portugal and Spain.",
                "Ember pilot results: over twelve weeks at the Lisbon depot, the on-time delivery rate rose from "
                "87 percent to 93 percent, and fuel cost per stop fell by 6 percent. Morning planning time for "
                "dispatchers fell from two hours to 25 minutes.",
            ],
            [
                "Decisions",
                "Ember will be rolled out to the Porto and Rotterdam depots in Q2 2026. Samuel Okafor will lead "
                "the rollout, and Alice Moreno moves to the redesign of the drivers' mobile app.",
                "Budget: the rollout is funded from the operations technology budget with no new headcount.",
            ],
        ],
    ),
    "corvid_contract_summary.pdf": (
        "Contract Summary: Corvid Data and Brightwater Health",
        [
            [
                "Contract summary: Corvid Data - Brightwater Health",
                "Service: address standardization and reference data for Project Lodestar.",
                "Term: 1 January 2026 to 31 December 2027. Fees: 180,000 dollars per year, billed quarterly.",
                "Service level: 99.5 percent monthly API availability. If availability falls below the SLA, "
                "Corvid credits 5 percent of the monthly fee for every 0.5 percent of shortfall.",
                "Data handling: Corvid processes addresses only and receives no clinical data. Contacts: Diego "
                "Alvarez for Corvid Data and Ravi Narayan for Brightwater Health.",
            ],
        ],
    ),
}


def render(filename: str, title: str, pages: list[list[str]]) -> None:
    pdf = FPDF(format="A4")
    pdf.set_title(title)
    pdf.set_author("ContextForge sample corpus")
    pdf.set_margins(20, 20, 20)
    for page in pages:
        pdf.add_page()
        for i, paragraph in enumerate(page):
            pdf.set_font("Helvetica", "B" if i == 0 else "", 14 if i == 0 else 11)
            pdf.multi_cell(0, 6, paragraph)
            pdf.ln(4)
    pdf.output(str(OUT / filename))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for filename, (title, pages) in DOCUMENTS.items():
        render(filename, title, pages)
        print(f"wrote sample_data/{filename}")


if __name__ == "__main__":
    main()
