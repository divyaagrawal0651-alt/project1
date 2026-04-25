# Bring in the Groq class from the groq library — this lets us talk to Groq's AI API.
from groq import Groq

# Bring in load_dotenv from the dotenv library — this reads secret keys from a .env file.
from dotenv import load_dotenv

# Bring in "os" — gives us tools to read environment variables (like API keys).
import os

# Call load_dotenv() right away — this reads the .env file in the project folder and
# loads all the key-value pairs (like GROQ_API_KEY=xxx) into the environment so we
# can access them later with os.getenv().
load_dotenv()


# Create a function called "ask_llm" that takes a prompt (a question/instruction string).
def ask_llm(prompt):
    """
    Big Picture:
    This function sends a prompt to the Groq AI API (using the Llama 3.1 8B model),
    waits for a response, and returns the AI's answer as a string. It acts as the
    "brain" of the RAG pipeline — once we have the relevant context from the documents,
    this function asks the AI to generate a human-readable answer.
    """

    # Create a Groq client object by passing in the API key.
    # os.getenv("GROQ_API_KEY") reads the value of GROQ_API_KEY from the environment.
    # Put the client into the variable "client".
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Use the client to create a chat completion (i.e., send a conversation to the AI
    # and get a response back). Put the result into "response".
    response = client.chat.completions.create(

        # Tell the API which AI model to use — "llama-3.1-8b-instant" (a fast Llama model).
        model="llama-3.1-8b-instant",

        # Send a list of messages (a conversation). This has two messages:
        messages=[
            {
                # The first message has the role "system" — this is an instruction to the AI
                # telling it how to behave.
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions "
                    "based only on the provided context. Be concise and "
                    "always cite the source document and page number."
                ),
            },

            # The second message has the role "user" — this is the actual question/prompt
            # that we want the AI to answer.
            {"role": "user", "content": prompt},
        ],

        # Set temperature to 0.1 — this makes the AI's answers very focused and consistent
        # (less random/creative). 0 = most deterministic, 1 = most creative.
        temperature=0.1,

        # Set the maximum number of tokens (words/pieces) the AI can use in its response to 1024.
        max_tokens=1024,
    )

    # From the response, go into choices → first choice [0] → message → content,
    # which is the actual text the AI generated. Return that text.
    return response.choices[0].message.content
