import base64
import sys
from pathlib import Path

from openai import OpenAI
from schemas import InvoiceData


EXTRACTION_PROMPT = """You are a document extraction specialist. Extract all invoice data from this image.

Rules:
- Extract only text values visible in the document.
- If a field is not found, set it to null.
- For each field, provide a confidence score between 0.0 and 1.0.
- For line items, extract description, quantity, unit_price, and amount.
- Amounts should be numeric values without currency symbols.
- Dates should be in the format found in the document (do not normalize).
"""

def encode_image(image_path: str) -> str:
    # Read the image file as bytes and encode to base64 string
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_invoice(image_path: str, client: OpenAI) -> InvoiceData:
    # Encode the image for the API
    base64_image = encode_image(image_path)

    # Detect media type from file extension
    file_ext = Path(image_path).suffix.lower()
    media_type = "image/png" if file_ext == ".png" else "image/jpeg"

    # Send multimodal request with Pydantic schema enforcement
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract all invoice data from this image."},
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <image_path>")
        sys.exit(1)

    client = OpenAI()
    result = extract_invoice(sys.argv[1], client)
    print(result.model_dump_json(indent=2))