from src.pdf_loader import extract_pages_from_pdf

with open("sample.pdf", "rb") as pdf:
    pages = extract_pages_from_pdf(pdf)

print(f"Total readable pages: {len(pages)}")

for page in pages:
    print("\n" + "=" * 40)
    print(f"Page: {page['page_number']}")
    print(f"Characters: {len(page['text'])}")
    print(page["text"][:150])