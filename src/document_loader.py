# Bring in PdfReader from the PyPDF2 library — this lets us read PDF files.
from PyPDF2 import PdfReader

# Bring in the "os" module — this gives us tools to work with file paths and file names.
import os


# Create a function called "load_document" that takes a file path (location of a file on disk).
def load_document(file_path):
    """
    Big Picture:
    This function reads a file (either PDF or TXT), pulls the text out of it,
    and returns it as a list of dictionaries. Each dictionary has the text content,
    the page number, and the file name. This is the first step in the RAG pipeline —
    getting raw text from your documents so the rest of the system can process it.
    """

    # Take the file path, split it into (name, extension), grab the extension part [1],
    # and make it lowercase. Put that into the variable "ext".
    # For example: "report.PDF" → ".pdf"
    ext = os.path.splitext(file_path)[1].lower()

    # From the full file path, grab just the file name (without the folder path)
    # and put it into the variable "name".
    # For example: "C:/docs/report.pdf" → "report.pdf"
    name = os.path.basename(file_path)

    # Check if the file extension is ".pdf"
    if ext == ".pdf":

        # Create a PdfReader object that opens and reads the PDF file,
        # and put it into the variable "reader".
        reader = PdfReader(file_path)

        # Create an empty list called "pages" to collect all the page data.
        pages = []

        # Loop through every page in the PDF. "enumerate" gives us both the
        # index number (i = 0, 1, 2...) and the page object itself.
        for i, page in enumerate(reader.pages):

            # Try to extract text from the current page. If extract_text()
            # returns nothing (None), use an empty string "" instead.
            # Put the result into the variable "text".
            text = page.extract_text() or ""

            # Check if the text has any real content (not just blank spaces).
            if text.strip():

                # If it does, create a dictionary with three things:
                #   - "text": the extracted text from this page
                #   - "page": the page number (i + 1 because i starts at 0, but pages start at 1)
                #   - "source": the file name
                # Then add it to the "pages" list.
                pages.append({"text": text, "page": i + 1, "source": name})

        # Give back (return) the list of all pages to whoever called this function.
        return pages

    # Check if the file extension is ".txt"
    if ext == ".txt":

        # Open the text file for reading ("r") with UTF-8 encoding,
        # and call the opened file "f".
        with open(file_path, "r", encoding="utf-8") as f:

            # Read the entire contents of the file and put it into the variable "text".
            text = f.read()

        # Return a list with one dictionary: the full text, page 1 (since .txt has no pages),
        # and the file name as the source.
        return [{"text": text, "page": 1, "source": name}]

    # If the file is neither .pdf nor .txt, raise an error telling the user
    # that this file type is not supported.
    raise ValueError(f"Unsupported file type: {ext}")
