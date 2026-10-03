
from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """Extract text from all pages of a PDF."""

    reader = PdfReader(pdf_file)
    extracted_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            extracted_text.append(text)

        print(f"Processed page {page_number}")

    return "\n\n".join(extracted_text)