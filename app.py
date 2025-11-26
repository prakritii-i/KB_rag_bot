import os
from dotenv import load_dotenv
from flask import Flask, request # For hosting
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

# RAG Libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Pinecone as LCPinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore # <-- ADD THIS
# Load environment variables (API keys)
load_dotenv()

# --- Configuration ---
INDEX_NAME = "slack-kb-index" # Must match the name used in ingest.py
EMBEDDING_MODEL = "models/embedding-001" # Gemini embedding model
LLM_MODEL = "gemini-1.5-flash" # The reliable, free LLM for reasoning

# --- Initialize RAG Components ---
try:
    # 1. Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 2. Initialize Gemini Embeddings and LLM
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # 3. Connect to the existing Pinecone index
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    print("RAG components initialized successfully.")

except Exception as e:
    print(f"FATAL ERROR: RAG initialization failed: {e}")
    # If this fails, the app won't start correctly.
    vectorstore = None
    llm = None


# --- Initialize Slack App and Flask Server ---

flask_app = Flask(__name__)
# The App class handles all Slack events, tokens, and secrets
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"), 
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)
handler = SlackRequestHandler(app)


# --- Slack Event Listener (The Core Logic) ---

@app.event("app_mention")
def handle_mentions(body, say):
    """
    Handles when the bot is mentioned in a Slack channel (e.g., @bot_name "What is HIPAA compliance?")
    """
    if not vectorstore or not llm:
        say("Error: The knowledge base is not connected. Please check the server logs.")
        return

    # 1. Immediate acknowledgement to avoid Slack timeout
    say(":mag: Searching internal documentation and reasoning...")
    
    # 2. Extract the user's query, removing the bot's mention tag
    query_text = body["event"]["text"].split('>', 1)[-1].strip()
    
    # 3. Retrieval (R) - Fetch relevant documents from Pinecone
    # We retrieve the top 3 most relevant document chunks
    docs = vectorstore.similarity_search(query_text, k=3)
    
    # 4. Augmentation (A) - Format the retrieved documents as context
    context = "\n---\n".join([f"Source: {d.metadata.get('source', 'Unknown File')}\nContent: {d.page_content}" for d in docs])
    
    # 5. Generation (G) - Craft the prompt for the LLM
    prompt = f"""
    You are an internal HIPAA compliance expert and a reliable assistant for a small team. 
    Your goal is to answer the user's question based ONLY on the provided context from the internal documents. 
    You must provide reasoning and reference the sources (e.g., 'Source: policies.pdf').
    
    If the context does not contain the answer, you must respond with: 
    'I apologize, I could not find a relevant answer in the internal knowledge base.'

    ---
    CONTEXT:
    {context}
    ---
    USER QUESTION: 
    {query_text}
    """
    
    # 6. Get the reasoned answer from the Gemini LLM
    try:
        response = llm.invoke(prompt)
        say(response.content)
    except Exception as e:
        say(f"I ran into an issue while generating the answer: {e}")


# --- Flask Endpoints for Hosting ---

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handles all incoming Slack events via the adapter."""
    return handler.handle(request)

@flask_app.route("/", methods=["GET"])
def health_check():
    """Endpoint for the Keep-Alive service (UptimeRobot)"""
    return "I am awake and ready to serve!", 200

# Entry point for the Render server (using Gunicorn)
if __name__ == "__main__":
    # Render will use Gunicorn to run this, so we don't need flask_app.run()
    print("Application is ready to be served by Gunicorn.")