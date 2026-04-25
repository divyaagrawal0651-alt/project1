# Create a function called "chunk_documents". It takes a list of pages,
# a chunk_size (default 500 characters), and a chunk_overlap (default 100 characters).
def chunk_documents(pages, chunk_size=1500, chunk_overlap=200):
    """
    Big Picture:
    This function takes a list of pages, slices each page's text into overlapping
    pieces of 500 characters (with 100 characters of overlap between consecutive
    pieces), tags each piece with its source file and page number, and returns all
    those pieces in a list.

    Why the overlap?
    When you split text into chunks, a sentence can get cut in half right at the
    boundary. By overlapping 100 characters, the end of one chunk and the start of
    the next chunk share some text, so no sentence is completely lost between two
    chunks. This is especially useful for search/RAG pipelines where you want each
    chunk to have enough context.
    """

    # Create an empty list called "chunks" — this is where we'll collect all our results.
    chunks = []

    # Go through each page in the pages list, one by one.
    for page in pages:

        # From the current page (which is a dictionary), grab the value stored
        # under the key "text" and put it into a variable called "text".
        text = page["text"]

        # Create a variable called "start" and set it to 0 — this is our reading
        # position, starting at the very beginning of the text.
        start = 0

        # Keep looping as long as our reading position (start) hasn't reached the end of the text.
        while start < len(text):

            # Calculate where this chunk should end: take the current start position
            # and add chunk_size (500) to it. Put that number into "end".
            end = start + chunk_size

            # From the full text, copy out the characters from position "start" up to
            # (but not including) position "end", and put that slice into "chunk_text".
            chunk_text = text[start:end]

            # Check if chunk_text has any real content (not just spaces/blanks).
            if chunk_text.strip():

                # If it does have real content, create a small dictionary with three things:
                #   - "text": the chunk of text we just cut out
                #   - "source": copy the source (file name) from the original page
                #   - "page": copy the page number from the original page
                # Then add (append) that dictionary to our "chunks" list.
                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                })

            # Move our reading position forward by (chunk_size - chunk_overlap),
            # which is 500 - 100 = 400. This means the next chunk will re-read
            # the last 100 characters of the current chunk, creating an overlap.
            start += chunk_size - chunk_overlap

    # After going through every page and every chunk inside each page,
    # give back (return) the full list of chunks to whoever called this function.
    return chunks
