
import docx
import PyPDF2

def read_txt(file_path):
    """
    Read .txt, .pdf, or .docx files and return the text content as a string.
    """
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)
    elif file_path.endswith(".pdf"):
        pdf_file = open(file_path, "rb")
        reader = PyPDF2.PdfReader(pdf_file)
        full_text = [page.extract_text() for page in reader.pages if page.extract_text()]
        pdf_file.close()
        return "\n".join(full_text)
    else:
        raise ValueError("Unsupported file type: " + file_path)
