import sys
from pathlib import Path

from openai import OpenAI
from schemas import ExtractionResult
from extractor import extract_invoice
from validator import validate_invoice
from qa_agent import run_qa_agent, calculate_overall_confidence


def print_report(result: ExtractionResult, image_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"EXTRACTION REPORT: {Path(image_path).name}")
    print(f"{'='*60}")

    invoice = result.invoice
    print(f"\n--- Extracted Data ---")
    print(f"Vendor:         {invoice.vendor_name or 'N/A'}")
    print(f"Invoice #:      {invoice.invoice_number or 'N/A'}")
    print(f"Date:           {invoice.invoice_date or 'N/A'}")
    print(f"Customer:       {invoice.customer_name or 'N/A'}")

    if invoice.line_items:
        print(f"\n--- Line Items ({len(invoice.line_items)}) ---")
        for i, item in enumerate(invoice.line_items, 1):
            qty = f"x{item.quantity}" if item.quantity else ""
            price = f"@ ${item.unit_price:.2f}" if item.unit_price else ""
            amount = f"= ${item.amount:.2f}" if item.amount else ""
            print(f"  {i}. {item.description} {qty} {price} {amount}")

    print(f"\n--- Totals ---")
    print(f"Subtotal:       ${invoice.subtotal:.2f}" if invoice.subtotal else "Subtotal:       N/A")
    print(f"Tax:            ${invoice.tax_amount:.2f}" if invoice.tax_amount else "Tax:            N/A")
    print(f"Total:          ${invoice.total_amount:.2f}" if invoice.total_amount else "Total:          N/A")

    print(f"\n--- Validation ---")
    for check in result.validation.checks:
        icon = "\u2705" if check.status == "pass" else "\u274c" if check.status == "fail" else "\u26a0\ufe0f"
        print(f"  {icon} {check.name}: {check.message}")

    print(f"\n--- QA Summary ---")
    print(f"Attempts:           {result.attempts}")
    print(f"Corrections Made:   {len(result.corrections_made)}")
    for correction in result.corrections_made:
        print(f"    - {correction}")
    print(f"Overall Confidence: {result.overall_confidence:.1%}")
    print(f"Status:             {'APPROVED' if result.overall_confidence >= 0.85 else 'NEEDS REVIEW'}")
    print(f"{'='*60}\n")


def process_invoice(image_path: str, client: OpenAI) -> ExtractionResult:
    print(f"\nProcessing: {image_path}")
    print("Step 1: Extracting data with vision model...")
    initial_extraction = extract_invoice(image_path, client)

    print("Step 2: Running validation checks...")
    validation = validate_invoice(initial_extraction)

    if validation.is_valid:
        print("Step 3: All checks passed! No QA correction needed.")
        overall = calculate_overall_confidence(initial_extraction, validation)
        result = ExtractionResult(
            invoice=initial_extraction,
            validation=validation,
            attempts=1,
            corrections_made=[],
            overall_confidence=overall,
        )
    else:
        print(f"Step 3: Issues found. Running QA agent (flagged: {', '.join(validation.flagged_fields)})...")
        result = run_qa_agent(image_path, initial_extraction, client)

    print_report(result, image_path)
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [image_path2 ...]")
        print("Example: python main.py invoices/sample1.png invoices/sample2.jpg")
        sys.exit(1)

    client = OpenAI()

    for image_path in sys.argv[1:]:
        if not Path(image_path).exists():
            print(f"Error: File not found: {image_path}")
            continue
        process_invoice(image_path, client)


if __name__ == "__main__":
    main()