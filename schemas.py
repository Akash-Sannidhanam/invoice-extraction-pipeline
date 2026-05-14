from typing import Optional
from pydantic import BaseModel


class InvoiceLineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    confidence: Optional[float] = None


class InvoiceData(BaseModel):
    vendor_name: Optional[str] = None
    vendor_name_confidence: Optional[float] = None
    vendor_address: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_number_confidence: Optional[float] = None
    invoice_date: Optional[str] = None
    invoice_date_confidence: Optional[float] = None
    due_date: Optional[str] = None
    customer_name: Optional[str] = None
    line_items: list[InvoiceLineItem] = []
    subtotal: Optional[float] = None
    subtotal_confidence: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_amount_confidence: Optional[float] = None
    total_amount: Optional[float] = None
    total_amount_confidence: Optional[float] = None
    currency: Optional[str] = None


class ValidationCheck(BaseModel):
    name: str
    status: str  # "pass", "fail", "warning"
    message: str


class ValidationResult(BaseModel):
    checks: list[ValidationCheck] = []
    is_valid: bool = True
    flagged_fields: list[str] = []


class ExtractionResult(BaseModel):
    invoice: InvoiceData
    validation: ValidationResult
    attempts: int = 1
    corrections_made: list[str] = []
    overall_confidence: float = 0.0