
# 🧾 Invoice Extraction Pipeline

An AI-powered document extraction dashboard that pulls structured data from invoice images, validates the results with deterministic checks, and automatically corrects its own mistakes using a self-correcting QA agent.

Built with OpenAI GPT-4o-mini, Streamlit, and Pydantic.

![Demo](demo-invoice.gif)


🔗 **[Try the live demo →](https://invoice-extraction-pipeline.streamlit.app/)**


## ✨ Features

| Feature | Description |
|---------|-------------|
| **AI Extraction** | GPT-4o-mini extracts vendor, line items, totals from invoice images |
| **Deterministic Validation** | Math checks, missing field detection, line item verification |
| **Self-Correcting QA Agent** | Automatically re-prompts the model with specific error feedback |
| **Streamlit Dashboard** | Drag-and-drop UI with two tabs: Single Invoice + Batch Processor |
| **PDF Support** | Multi-page PDFs split into individual pages for extraction |
| **Confidence Indicators** | 🟢🟡🔴 per-field confidence badges and bar charts |
| **Adjustable Threshold** | Sidebar slider to tune approval strictness in real time |
| **Processing Timer** | Shows extraction time per invoice |
| **CSV Export** | Download extracted data as a spreadsheet-ready CSV |
| **Markdown Reports** | Downloadable batch reports with review queues |
| **Batch Processing** | Process entire folders of invoices with aggregate metrics |

## 🏗️ Architecture

```
Invoice Image → GPT-4o-mini Extraction → Deterministic Validation
                                              ↓
                                         Pass? → Done ✅
                                         Fail? → QA Agent corrects → Re-validate (max 2 retries)
```

**Key modules:**
- `schemas.py` — Pydantic models for structured extraction
- `extractor.py` — Vision-based invoice extraction with OpenAI
- `validator.py` — Deterministic validation checks (math, missing fields)
- `qa_agent.py` — Self-correcting QA agent with confidence scoring
- `batch.py` — Batch processing and markdown report generation
- `app.py` — Streamlit dashboard with all UI features

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Poppler](https://poppler.freedesktop.org/) (for PDF support)
- OpenAI API key

### Setup

```bash
# Clone the repo
git clone https://github.com/Akash-Sannidhanam/invoice-extraction-pipeline.git
cd invoice-extraction-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Poppler (macOS)
brew install poppler

# Add your API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Run from CLI

```bash
# Single invoice
python main.py invoices/sample1.png

# Multiple invoices
python main.py invoices/sample1.png invoices/sample2.png
```

### Run Batch Processor

```bash
python batch.py invoices
```

## 📊 How It Works

1. **Extract** — GPT-4o-mini reads the invoice image and returns structured JSON via OpenAI Structured Outputs
2. **Validate** — Deterministic checks verify math (subtotal + tax = total), detect missing fields, and validate line items
3. **Correct** — If validation fails, the QA agent constructs a targeted correction prompt with specific error messages, re-sends the image, and re-validates (max 2 retries)
4. **Score** — Overall confidence combines model self-reported confidence (60%) with validation pass rate (40%)
5. **Approve** — Documents above the threshold are auto-approved; others go to a review queue

## 🛠️ Built With

- [OpenAI API](https://platform.openai.com/) — GPT-4o-mini for vision-based extraction
- [Streamlit](https://streamlit.io/) — Interactive dashboard UI
- [Pydantic](https://docs.pydantic.dev/) — Schema validation and structured outputs
- [pdf2image](https://github.com/Belval/pdf2image) — PDF to image conversion
- [Pandas](https://pandas.pydata.org/) — Data display and CSV export

## 📁 Project Structure

```
├── app.py                 # Streamlit dashboard
├── main.py                # CLI entry point
├── batch.py               # Batch processing + report generation
├── extractor.py           # Vision-based invoice extraction
├── validator.py           # Deterministic validation checks
├── qa_agent.py            # Self-correcting QA agent
├── schemas.py             # Pydantic data models
├── requirements.txt       # Python dependencies
├── .env                   # API keys (git-ignored)
├── .gitignore             # Files excluded from Git
└── invoices/              # Sample invoice images
```

## 📄 License

This project was built as part of a [NextWork](https://learn.nextwork.org) hands-on project.
