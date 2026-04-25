# From our embeddings module, bring in the embed_texts function
# (converts text into numerical vectors).
from src.embeddings import embed_texts

# From our vector_store module, bring in the search function
# (finds the most similar chunks in the saved data).
from src.vector_store import search

# From our llm module, bring in the ask_llm function
# (sends a prompt to the AI and gets an answer back).
from src.llm import ask_llm


# Create a function called "answer_question" that takes a question (string)
# and an optional top_k (how many matching chunks to find, default 3).
def answer_question(question, top_k=5):
    """
    Big Picture:
    This is the main RAG (Retrieval-Augmented Generation) function that ties
    everything together. It takes a user's question, converts it into a vector,
    searches for the most relevant document chunks, builds a prompt with those
    chunks as context, sends it to the AI, and returns the AI's answer along
    with the source chunks. This is the full pipeline:
    Question → Embedding → Search → Build Prompt → AI Answer.
    """

    # Take the user's question, put it in a list [question], convert it into an embedding
    # vector using embed_texts, then grab the first (and only) result [0].
    # Put that single vector into "query_embedding".
    query_embedding = embed_texts([question])[0]

    # Search the vector store for the top_k chunks that are most similar to the query_embedding.
    # Put the list of matching chunks into "chunks".
    chunks = search(query_embedding, top_k=top_k)

    # If no chunks were found (the list is empty), it means no documents have been uploaded yet.
    if not chunks:
        # Return a message telling the user to upload documents, and an empty list for chunks.
        return "No documents found. Please upload some documents first.", []

    # Build a "context" string by joining all the matching chunks together.
    # For each chunk "c", create a line like: [Source: filename.pdf, Page 3]
    # followed by the chunk's text. Separate each chunk with two blank lines (\n\n).
    context = "\n\n".join(
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    )

    # Build the full prompt that we'll send to the AI. It includes:
    #   - An instruction: only answer from the context, say "I don't know" if not enough info
    #   - The context (all the relevant chunks we found)
    #   - The user's original question
    #   - A reminder to cite the source and page number
    prompt = (
        "Answer the question based ONLY on the context below. "
        "If the context doesn't contain the answer, say "
        '"I don\'t have enough information to answer this."\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Cite the source document name and page number in your answer."
    )

    # Send the prompt to the AI using ask_llm and put the AI's response into "answer".
    answer = ask_llm(prompt)

    # Return two things: the AI's answer (a string) and the list of chunks that were used
    # (so the caller can show the user which sources were referenced).
    return answer, chunks
