import base64
from pathlib import Path

from openai import OpenAI
from schemas import ExtractionResult, InvoiceData, ValidationResult
from extractor import encode_image
from validator import validate_invoice


CORRECTION_PROMPT_TEMPLATE = """You are a document extraction specialist performing a quality review.

A previous extraction attempt produced errors. Here are the specific issues found:

{issues}

Please re-examine the invoice image carefully and provide a corrected extraction.
Pay special attention to the flagged fields. Common errors include:
- Misreading digits (e.g., 8 vs 6, 1 vs 7)
- Incorrect decimal placement
- Missing line items
- Swapping subtotal and total values

Rules:
- Extract only text values visible in the document.
- If a field is not found, set it to null.
- For each field, provide a confidence score between 0.0 and 1.0.
- Amounts should be numeric values without currency symbols.
"""


def calculate_overall_confidence(invoice: InvoiceData, validation: ValidationResult) -> float:
    confidence_values = []

    # Collect all field-level confidence scores reported by the model
    field_confidences = [
        invoice.vendor_name_confidence,
        invoice.invoice_number_confidence,
        invoice.invoice_date_confidence,
        invoice.subtotal_confidence,
        invoice.tax_amount_confidence,
        invoice.total_amount_confidence,
    ]
    for conf in field_confidences:
        if conf is not None:
            confidence_values.append(conf)

    # Default to 0.5 if no confidence values were reported
    if not confidence_values:
        field_avg = 0.5
    else:
        field_avg = sum(confidence_values) / len(confidence_values)

    # Calculate what percentage of validation checks passed
    total_checks = len(validation.checks)
    passed_checks = sum(1 for c in validation.checks if c.status == "pass")
    validation_score = passed_checks / total_checks if total_checks > 0 else 0.5

    # Weighted combination: 60% model confidence, 40% validation pass rate
    overall = (field_avg * 0.6) + (validation_score * 0.4)
    return round(overall, 3)


def attempt_correction(image_path: str, validation: ValidationResult, client: OpenAI,) -> InvoiceData:
    # Format failed checks into a bullet list for the correction prompt
    issues = "\n".join(
        f"- [{check.name}] {check.message}"
        for check in validation.checks
        if check.status == "fail"
    )

    base64_image = encode_image(image_path)
    file_ext = Path(image_path).suffix.lower()
    media_type = "image/png" if file_ext == ".png" else "image/jpeg"

    # Insert the specific issues into the correction prompt template
    prompt = CORRECTION_PROMPT_TEMPLATE.format(issues=issues)

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Re-extract the invoice data, correcting the issues listed above."},
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{base64_image}",
                    },
                ],
            },
        ],
        text_format=InvoiceData,
    )

    return response.output_parsed


def run_qa_agent(image_path: str, initial_extraction: InvoiceData, client: OpenAI, max_retries: int = 2,) -> ExtractionResult:
    current_extraction = initial_extraction
    validation = validate_invoice(current_extraction)
    attempts = 1
    corrections_made = []

    # Loop until validation passes or we hit the retry limit
    while not validation.is_valid and attempts <= max_retries:
        attempts += 1
        corrections_made.append(
            f"Attempt {attempts}: Correcting {', '.join(validation.flagged_fields)}"
        )

        # Re-extract with targeted correction prompt
        current_extraction = attempt_correction(image_path, validation, client)
        validation = validate_invoice(current_extraction)

    # Score the final result regardless of whether all checks passed
    overall_confidence = calculate_overall_confidence(current_extraction, validation)

    return ExtractionResult(
        invoice=current_extraction,
        validation=validation,
        attempts=attempts,
        corrections_made=corrections_made,
        overall_confidence=overall_confidence,
    )