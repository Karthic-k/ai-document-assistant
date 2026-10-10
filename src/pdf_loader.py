
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



def extract_pages_from_pdf(pdf_file):
    """
    Extract text separately from each PDF page,
    preserving page numbers.
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

    return pages