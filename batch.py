import sys
from pathlib import Path
from collections import Counter

from openai import OpenAI
from extractor import extract_invoice
from validator import validate_invoice
from qa_agent import run_qa_agent, calculate_overall_confidence
from schemas import ExtractionResult


def process_batch(directory: str, client: OpenAI) -> list[dict]:
    """Process all invoice images in a directory."""
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    invoice_dir = Path(directory)
    results = []

    image_files = [f for f in invoice_dir.iterdir() if f.suffix.lower() in image_extensions]
    if not image_files:
        print(f"No image files found in {directory}")
        return results

    for image_path in sorted(image_files):
        print(f"Processing: {image_path.name}...")
        initial_extraction = extract_invoice(str(image_path), client)
        validation = validate_invoice(initial_extraction)

        if validation.is_valid:
            overall = calculate_overall_confidence(initial_extraction, validation)
            result = ExtractionResult(
                invoice=initial_extraction,
                validation=validation,
                attempts=1,
                corrections_made=[],
                overall_confidence=overall,
            )
        else:
            result = run_qa_agent(str(image_path), initial_extraction, client)

        results.append({"file": image_path.name, "result": result})

    return results


def generate_report(results: list[dict]) -> str:
    """Generate a markdown report with aggregate metrics and review queue."""
    report_lines = ["# Batch Extraction Report\n"]

    # Calculate aggregate metrics
    confidences = [r["result"].overall_confidence for r in results]
    total_corrections = sum(len(r["result"].corrections_made) for r in results)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    report_lines.append("## Summary\n")
    report_lines.append(f"- **Documents processed**: {len(results)}")
    report_lines.append(f"- **Average confidence**: {avg_confidence:.1%}")
    report_lines.append(f"- **Total corrections made**: {total_corrections}")
    report_lines.append(f"- **Documents needing review**: {sum(1 for c in confidences if c < 0.85)}\n")

    # Per-document results table
    report_lines.append("## Per-Document Results\n")
    report_lines.append("| Document | Confidence | Attempts | Status |")
    report_lines.append("|----------|-----------|----------|--------|")
    for r in results:
        result = r["result"]
        status = "APPROVED" if result.overall_confidence >= 0.85 else "NEEDS REVIEW"
        report_lines.append(
            f"| {r['file']} | {result.overall_confidence:.1%} | {result.attempts} | {status} |"
        )

    # Review queue for documents below the confidence threshold
    review_queue = [r for r in results if r["result"].overall_confidence < 0.85]
    if review_queue:
        report_lines.append("\n## Review Queue\n")
        report_lines.append("These documents scored below 85% confidence and need human review:\n")
        for r in sorted(review_queue, key=lambda x: x["result"].overall_confidence):
            result = r["result"]
            report_lines.append(f"### {r['file']} ({result.overall_confidence:.1%})\n")
            for check in result.validation.checks:
                if check.status == "fail":
                    report_lines.append(f"- {check.name}: {check.message}")

    # Most commonly flagged fields across all documents
    all_flagged = []
    for r in results:
        all_flagged.extend(r["result"].validation.flagged_fields)
    if all_flagged:
        report_lines.append("\n## Most Commonly Flagged Fields\n")
        field_counts = Counter(all_flagged).most_common()
        for field, count in field_counts:
            report_lines.append(f"- **{field}**: flagged {count} time(s)")

    return "\n".join(report_lines)


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "invoices"
    if not Path(directory).exists():
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)

    client = OpenAI()
    print(f"Processing all invoices in '{directory}'...\n")

    results = process_batch(directory, client)
    if results:
        report = generate_report(results)
        report_path = Path("batch_report.md")
        report_path.write_text(report)
        print(f"\nReport saved to: {report_path}")
        print(report)


if __name__ == "__main__":
    main()