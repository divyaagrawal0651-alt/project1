# Bring in "faiss" — Facebook's library for fast similarity search on vectors.
import faiss

# Bring in "numpy" (nicknamed "np") — a library for working with arrays of numbers.
import numpy as np

# Bring in "pickle" — a built-in Python tool for saving and loading Python objects to/from files.
import pickle

# Bring in "os" — gives us tools for working with file paths and folders.
import os

# Build the path to the storage folder: "data/store". Put it in STORE_DIR.
STORE_DIR = os.path.join("data", "store")

# Build the full path to the FAISS index file: "data/store/faiss.index". Put it in INDEX_PATH.
INDEX_PATH = os.path.join(STORE_DIR, "faiss.index")

# Build the full path to the metadata file: "data/store/chunks.pkl". Put it in META_PATH.
META_PATH = os.path.join(STORE_DIR, "chunks.pkl")


# A small helper function that creates the storage folder if it doesn't already exist.
def _ensure_dir():
    """Makes sure the 'data/store' folder exists. If not, it creates it."""

    # Create the STORE_DIR folder (and any parent folders). If it already exists, do nothing.
    os.makedirs(STORE_DIR, exist_ok=True)


# Create a function that adds new chunks and their embeddings to the vector store.
def add_chunks(new_chunks, new_embeddings):
    """
    Big Picture:
    This function saves new document chunks and their embeddings into a FAISS index
    (for fast searching) and a pickle file (for storing the chunk metadata like text,
    source, and page number). If there's already saved data, it loads it first and
    adds the new data on top. This is how the system "remembers" your uploaded documents.
    """

    # Make sure the storage folder exists.
    _ensure_dir()

    # Convert the new embeddings to 32-bit floating point numbers (FAISS requires this format).
    # Put the result into the variable "embeddings".
    embeddings = new_embeddings.astype(np.float32)

    # Check if a FAISS index file already exists on disk (meaning we've stored data before).
    if os.path.exists(INDEX_PATH):

        # If yes, read (load) the existing FAISS index from disk into the variable "index".
        index = faiss.read_index(INDEX_PATH)

        # Also open the metadata file in read-binary mode ("rb"),
        # load the saved list of chunks from it, and put it into "chunks".
        with open(META_PATH, "rb") as f:
            chunks = pickle.load(f)

    else:
        # If no existing index, figure out the dimension (size) of each embedding vector.
        # embeddings.shape[1] gives us the number of columns (i.e., the vector length).
        dim = embeddings.shape[1]

        # Create a brand new FAISS index that uses L2 (Euclidean) distance for comparing vectors.
        # The "dim" tells FAISS how long each vector is.
        index = faiss.IndexFlatL2(dim)

        # Start with an empty list of chunks since we have no previous data.
        chunks = []

    # Add the new embedding vectors into the FAISS index.
    index.add(embeddings)

    # Add the new chunk dictionaries to the end of the chunks list.
    chunks.extend(new_chunks)

    # Save (write) the updated FAISS index back to disk.
    faiss.write_index(index, INDEX_PATH)

    # Open the metadata file in write-binary mode ("wb") and
    # save (dump) the updated chunks list into it using pickle.
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)


# Create a function that searches the vector store for chunks similar to a query.
def search(query_embedding, top_k=3):
    """
    Big Picture:
    This function takes a query embedding (the numerical vector of your question),
    searches the FAISS index for the most similar chunk vectors, and returns the
    top_k (default 3) most relevant chunks. This is how the system finds which
    parts of your documents are most related to your question.
    """

    # If no FAISS index file exists on disk, there's nothing to search — return an empty list.
    if not os.path.exists(INDEX_PATH):
        return []

    # Load the FAISS index from disk into the variable "index".
    index = faiss.read_index(INDEX_PATH)

    # Open the metadata file, load the chunks list, and put it into "chunks".
    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Wrap the query_embedding in a list (to make it 2D), convert to float32,
    # and put it into "query_vec". FAISS expects a 2D array even for a single query.
    query_vec = np.array([query_embedding]).astype(np.float32)

    # Ask FAISS to search for the top_k nearest vectors to our query_vec.
    # It returns two things: distances (which we ignore with "_") and indices
    # (the positions of the matching chunks). Put the indices into "indices".
    _, indices = index.search(query_vec, top_k)

    # Go through the indices from the first (and only) query result [0],
    # and for each valid index i (between 0 and the length of chunks),
    # grab the corresponding chunk from the chunks list.
    # Return this list of matching chunks.
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


# Create a function that checks whether a FAISS index exists on disk.
def has_index():
    """Returns True if there is saved data (a FAISS index file exists), otherwise False."""

    # Check if the index file exists at INDEX_PATH and return True or False.
    return os.path.exists(INDEX_PATH)


# Create a function that deletes all saved data (clears the vector store).
def clear_store():
    """
    Big Picture:
    This function removes both the FAISS index file and the chunks metadata file
    from disk, effectively wiping all stored document data. Use this when you
    want to start fresh.
    """

    # Go through both file paths (the index file and the metadata file).
    for path in [INDEX_PATH, META_PATH]:

        # If the file exists at that path, delete it.
        if os.path.exists(path):
            os.remove(path)
