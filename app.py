
import streamlit as st
import tempfile
import pandas as pd
from pathlib import Path

from openai import OpenAI
from pdf2image import convert_from_path
from extractor import extract_invoice
from validator import validate_invoice
from qa_agent import run_qa_agent, calculate_overall_confidence
from schemas import ExtractionResult
from batch import process_batch, generate_report

from dotenv import load_dotenv
load_dotenv()



# ── Helper Functions ──

def save_upload_as_image(uploaded_file, tmp_dir: str) -> list[str]:
    """Saves an uploaded file to tmp_dir. Converts PDFs to PNGs. Returns list of image paths."""
    file_ext = Path(uploaded_file.name).suffix.lower()

    if file_ext == ".pdf":
        pdf_path = Path(tmp_dir) / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.read())

        pages = convert_from_path(str(pdf_path))
        image_paths = []
        for i, page in enumerate(pages):
            img_path = Path(tmp_dir) / f"{pdf_path.stem}_page{i + 1}.png"
            page.save(str(img_path), "PNG")
            image_paths.append(str(img_path))
        return image_paths
    else:
        img_path = Path(tmp_dir) / uploaded_file.name
        img_path.write_bytes(uploaded_file.read())
        return [str(img_path)]


def confidence_badge(value):
    """Returns a colored badge based on confidence score."""
    if value is None:
        return "⚪ N/A"
    if value >= 0.9:
        return f"🟢 {value:.0%}"
    elif value >= 0.7:
        return f"🟡 {value:.0%}"
    else:
        return f"🔴 {value:.0%}"


def display_invoice(result: ExtractionResult):
    """Renders extracted invoice data with confidence indicators."""
    invoice = result.invoice

    # ── Key fields with confidence ──
    st.markdown("#### Invoice Details")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown(f"**Vendor:** {invoice.vendor_name or 'N/A'} {confidence_badge(invoice.vendor_name_confidence)}")
        st.markdown(f"**Invoice #:** {invoice.invoice_number or 'N/A'} {confidence_badge(invoice.invoice_number_confidence)}")
    with detail_col2:
        st.markdown(f"**Date:** {invoice.invoice_date or 'N/A'} {confidence_badge(invoice.invoice_date_confidence)}")
        st.markdown(f"**Customer:** {invoice.customer_name or 'N/A'}")

    # ── Line items as a table ──
    if invoice.line_items:
        st.markdown("#### Line Items")
        rows = []
        for item in invoice.line_items:
            rows.append({
                "Description": item.description,
                "Qty": item.quantity or "—",
                "Unit Price": f"${item.unit_price:.2f}" if item.unit_price else "—",
                "Amount": f"${item.amount:.2f}" if item.amount else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Totals with confidence ──
    st.markdown("#### Totals")
    total_col1, total_col2, total_col3 = st.columns(3)
    total_col1.metric("Subtotal", f"${invoice.subtotal:.2f}" if invoice.subtotal else "N/A")
    total_col1.caption(confidence_badge(invoice.subtotal_confidence))
    total_col2.metric("Tax", f"${invoice.tax_amount:.2f}" if invoice.tax_amount else "N/A")
    total_col2.caption(confidence_badge(invoice.tax_amount_confidence))
    total_col3.metric("Total", f"${invoice.total_amount:.2f}" if invoice.total_amount else "N/A")
    total_col3.caption(confidence_badge(invoice.total_amount_confidence))

    # ── Confidence legend ──
    st.caption("🟢 High (90%+)  🟡 Medium (70-89%)  🔴 Low (<70%)")


def display_validation(result: ExtractionResult):
    """Renders validation checks, confidence, status, and QA corrections."""
    st.subheader("Validation Results")
    for check in result.validation.checks:
        if check.status == "pass":
            st.success(f"{check.name}: {check.message}")
        elif check.status == "fail":
            st.error(f"{check.name}: {check.message}")
        else:
            st.warning(f"{check.name}: {check.message}")

    conf_col, status_col = st.columns(2)
    conf_col.metric("Overall Confidence", f"{result.overall_confidence:.1%}")
    if result.overall_confidence >= 0.85:
        status_col.success("✅ APPROVED")
    else:
        status_col.error("⚠️ NEEDS REVIEW")

    if result.corrections_made:
        st.subheader("QA Agent Corrections")
        for correction in result.corrections_made:
            st.info(correction)


# ── Page Config ──

st.set_page_config(page_title="Invoice Extraction Pipeline", layout="wide")
st.title("Invoice Extraction Pipeline")

tab1, tab2 = st.tabs(["Single Invoice", "Batch Processor"])


# ── Tab 1: Single Invoice ──

with tab1:
    st.write("Upload an invoice image or PDF to extract structured data with AI-powered QA.")
    uploaded_file = st.file_uploader(
        "Upload an invoice image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        key="single_uploader",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Invoice", width=300)

        if st.button("Extract Invoice", key="single_extract"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_paths = save_upload_as_image(uploaded_file, tmp_dir)

                all_results = []
                for img_path in image_paths:
                    with st.spinner(f"Extracting page {image_paths.index(img_path) + 1} of {len(image_paths)}..."):
                        client = OpenAI()
                        initial_extraction = extract_invoice(img_path, client)
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
                            with st.spinner("Issues found. Running QA agent..."):
                                result = run_qa_agent(img_path, initial_extraction, client)

                        all_results.append(result)

            if len(all_results) == 1:
                st.session_state["single_result"] = all_results[0]
            else:
                st.session_state["single_result"] = all_results

    # Display stored results
    if "single_result" in st.session_state:
        stored = st.session_state["single_result"]

        # Handle multi-page PDFs
        if isinstance(stored, list):
            for i, result in enumerate(stored):
                st.markdown(f"---")
                st.subheader(f"Page {i + 1}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("Uploaded Invoice")
                    if uploaded_file is not None:
                        st.image(uploaded_file, use_container_width=True)
                with col2:
                    st.subheader("Extraction Results")
                    display_invoice(result)
                display_validation(result)
        else:
            result = stored
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Uploaded Invoice")
                if uploaded_file is not None:
                    st.image(uploaded_file, use_container_width=True)
            with col2:
                st.subheader("Extraction Results")
                display_invoice(result)
            display_validation(result)


# ── Tab 2: Batch Processor ──

with tab2:
    st.write("Upload multiple invoice images or PDFs to process them as a batch.")
    uploaded_files = st.file_uploader(
        "Upload invoice images or PDFs",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if uploaded_files and st.button("Run Batch Processing", key="batch_extract"):
        client = OpenAI()

        image_data = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            for uf in uploaded_files:
                file_bytes = uf.read()
                image_paths = save_upload_as_image(uf, tmp_dir)
                for img_path in image_paths:
                    image_data[Path(img_path).name] = Path(img_path).read_bytes()

            with st.spinner(f"Processing {len(uploaded_files)} invoices..."):
                results = process_batch(tmp_dir, client)

        st.session_state["batch_results"] = results
        st.session_state["batch_images"] = image_data

    # Display stored results
    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]

        # ── Summary metrics ──
        st.subheader("Batch Summary")
        confidences = [r["result"].overall_confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        needs_review = sum(1 for c in confidences if c < 0.85)

        col1, col2, col3 = st.columns(3)
        col1.metric("Documents Processed", len(results))
        col2.metric("Avg Confidence", f"{avg_confidence:.1%}")
        col3.metric("Needs Review", needs_review)

        # ── Per-document results ──
        st.subheader("Per-Document Results")
        for r in results:
            res = r["result"]
            status = "APPROVED" if res.overall_confidence >= 0.85 else "NEEDS REVIEW"
            icon = "✅" if status == "APPROVED" else "⚠️"
            with st.expander(f"{icon} {r['file']} — {res.overall_confidence:.1%} ({status})"):
                img_col, data_col = st.columns([1, 2])
                with img_col:
                    if r["file"] in st.session_state.get("batch_images", {}):
                        st.image(
                            st.session_state["batch_images"][r["file"]],
                            caption=r["file"],
                            use_container_width=True,
                        )
                with data_col:
                    display_invoice(res)

                for check in res.validation.checks:
                    if check.status == "pass":
                        st.success(f"{check.name}: {check.message}")
                    elif check.status == "fail":
                        st.error(f"{check.name}: {check.message}")
                    else:
                        st.warning(f"{check.name}: {check.message}")
                if res.corrections_made:
                    for correction in res.corrections_made:
                        st.info(correction)

        # ── Downloadable report ──
        report = generate_report(results)
        st.download_button("Download Batch Report", report, "batch_report.md", "text/markdown")
