# Bring in SentenceTransformer from the sentence_transformers library —
# this lets us convert text into numerical vectors (embeddings).
from sentence_transformers import SentenceTransformer

# Create a variable called "_model" and set it to None.
# This will hold our loaded model later. The underscore means "private / internal use".
_model = None


# Create a function called "get_model" that loads the embedding model.
def get_model():
    """
    Big Picture:
    This function loads the sentence transformer model into memory. It uses a trick
    called "lazy loading" — the model is only loaded the first time you call this
    function. After that, it reuses the already-loaded model (stored in _model) so
    we don't waste time loading it again every single time.
    """

    # Tell Python we want to use the _model variable that lives outside this function (global).
    global _model

    # Check if the model has NOT been loaded yet (still None).
    if _model is None:

        # Load the "all-MiniLM-L6-v2" model (a small, fast embedding model)
        # and store it in the _model variable so we can reuse it later.
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    # Give back (return) the loaded model.
    return _model


# Create a function called "embed_texts" that takes a list of text strings.
def embed_texts(texts):
    """
    Big Picture:
    This function takes a list of text strings, converts each one into a numerical
    vector (a list of numbers) using the embedding model, and returns all those
    vectors. These vectors capture the "meaning" of each text, so similar texts
    will have similar vectors. This is how the system can later find relevant
    chunks when you ask a question.
    """

    # Call get_model() to get the loaded embedding model, put it into "model".
    model = get_model()

    # Use the model to convert (encode) all the texts into numerical vectors.
    # show_progress_bar=False means don't print a loading bar in the console.
    # Return the resulting vectors.
    return model.encode(texts, show_progress_bar=False)
