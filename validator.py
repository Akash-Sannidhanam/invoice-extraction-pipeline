from schemas import InvoiceData, ValidationCheck, ValidationResult


def validate_line_items_sum(invoice: InvoiceData) -> ValidationCheck:
    # If line items or subtotal are missing, we can't verify
    if not invoice.line_items or invoice.subtotal is None:
        return ValidationCheck(
            name="line_items_sum",
            status="warning",
            message="Cannot verify: missing line items or subtotal",
        )

    # Sum all line item amounts, skipping any that are None
    calculated_sum = sum(
        item.amount for item in invoice.line_items if item.amount is not None
    )
    # Allow a small rounding tolerance for floating point math
    tolerance = 0.02

    if abs(calculated_sum - invoice.subtotal) <= tolerance:
        return ValidationCheck(
            name="line_items_sum",
            status="pass",
            message=f"Line items sum ({calculated_sum:.2f}) matches subtotal ({invoice.subtotal:.2f})",
        )
    else:
        return ValidationCheck(
            name="line_items_sum",
            status="fail",
            message=f"Line items sum ({calculated_sum:.2f}) does not match subtotal ({invoice.subtotal:.2f})",
        )


def validate_total(invoice: InvoiceData) -> ValidationCheck:
    # Need both subtotal and total to run this check
    if invoice.subtotal is None or invoice.total_amount is None:
        return ValidationCheck(
            name="total_check",
            status="warning",
            message="Cannot verify: missing subtotal or total",
        )

    # If tax is not extracted, assume zero
    tax = invoice.tax_amount if invoice.tax_amount is not None else 0.0
    expected_total = invoice.subtotal + tax
    tolerance = 0.02

    if abs(expected_total - invoice.total_amount) <= tolerance:
        return ValidationCheck(
            name="total_check",
            status="pass",
            message=f"Subtotal ({invoice.subtotal:.2f}) + tax ({tax:.2f}) = total ({invoice.total_amount:.2f})",
        )
    else:
        return ValidationCheck(
            name="total_check",
            status="fail",
            message=f"Subtotal ({invoice.subtotal:.2f}) + tax ({tax:.2f}) = {expected_total:.2f}, but total is {invoice.total_amount:.2f}",
        )


def validate_required_fields(invoice: InvoiceData) -> ValidationCheck:
    # These fields must be present for a usable extraction
    required = ["vendor_name", "invoice_number", "invoice_date", "total_amount"]
    missing = [f for f in required if getattr(invoice, f) is None]

    if not missing:
        return ValidationCheck(
            name="required_fields",
            status="pass",
            message="All required fields present",
        )
    else:
        return ValidationCheck(
            name="required_fields",
            status="fail",
            message=f"Missing required fields: {', '.join(missing)}",
        )


def validate_confidence(invoice: InvoiceData) -> ValidationCheck:
    # Check confidence scores for key fields
    confidence_fields = [
        ("vendor_name", invoice.vendor_name_confidence),
        ("invoice_number", invoice.invoice_number_confidence),
        ("invoice_date", invoice.invoice_date_confidence),
        ("total_amount", invoice.total_amount_confidence),
    ]
    # Flag any field with confidence below 85%
    low_confidence = [
        name for name, conf in confidence_fields if conf is not None and conf < 0.85
    ]

    if not low_confidence:
        return ValidationCheck(
            name="confidence_check",
            status="pass",
            message="All key fields have high confidence",
        )
    else:
        return ValidationCheck(
            name="confidence_check",
            status="warning",
            message=f"Low confidence on: {', '.join(low_confidence)}",
        )


def validate_invoice(invoice: InvoiceData) -> ValidationResult:
    # Run all validation checks
    checks = [
        validate_line_items_sum(invoice),
        validate_total(invoice),
        validate_required_fields(invoice),
        validate_confidence(invoice),
    ]

    flagged_fields = []
    is_valid = True

    for check in checks:
        if check.status == "fail":
            is_valid = False
            # Map check failures to specific field names
            if "line_items" in check.name:
                flagged_fields.append("subtotal")
            elif "total" in check.name:
                flagged_fields.append("total_amount")
            elif "required" in check.name:
                flagged_fields.extend(
                    check.message.replace("Missing required fields: ", "").split(", ")
                )
        elif check.status == "warning" and "confidence" in check.name:
            # Low confidence fields get flagged but don't fail validation
            low_fields = check.message.replace("Low confidence on: ", "").split(", ")
            flagged_fields.extend(low_fields)

    return ValidationResult(
        checks=checks,
        is_valid=is_valid,
        flagged_fields=flagged_fields,
    )