
from src.pdf_loader import extract_text_from_pdf

pdf_path = "sample.pdf"

with open(pdf_path, "rb") as pdf_file:
    text = extract_text_from_pdf(pdf_file)

print("\n--- EXTRACTED TEXT ---\n")
print(text)

print("\n--- SUMMARY ---")
print("Characters extracted:", len(text))