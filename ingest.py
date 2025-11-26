import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LCPinecone

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
INDEX_NAME = "slack-kb-index"
DOCS_PATH = "docs"

# --- 1. Initialize Pinecone and Embeddings ---

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize Gemini Embeddings
# We use the free embedding model for this task.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 2. Create Pinecone Index (if it doesn't exist) ---
def create_pinecone_index():
    print(f"Checking for existing index: {INDEX_NAME}...")
    
    if INDEX_NAME not in pc.list_indexes().names:
        print(f"Index '{INDEX_NAME}' not found. Creating a new one...")
        
        # NOTE: For the free tier, we must use ServerlessSpec or the default environment
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, # Dimension size for the embedding-001 model
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Use default region if needed
        )
        print("Index created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists. Proceeding...")

# --- 3. Load and Split Documents ---
def load_and_split_documents():
    print(f"Loading documents from the '{DOCS_PATH}' directory...")
    
    # We use a DirectoryLoader to handle all files in the docs folder
    # For PDFs, you'll need to use the PyPDFLoader in a more complex setup, 
    # but for simplicity, we load all as text initially.
    # If you have PDFs, you must install 'pypdf' and specify the loader:
    # loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    # For now, let's load all files as text (TXT, MD, etc.)
    loader = DirectoryLoader(
        DOCS_PATH, 
        glob="**/*.*", 
        loader_cls=TextLoader, 
        loader_kwargs={'autodetect_encoding': True}
    )
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents.")

    # Split documents into smaller chunks (optimal size for RAG)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    print(f"Split into {len(texts)} text chunks.")
    return texts

# --- 4. Embed and Store ---
def embed_and_store(texts):
    print("Embedding chunks and storing in Pinecone...")
    
    # This uses LangChain's utility to simplify the embedding and uploading process
    LCPinecone.from_documents(
        texts,
        embeddings,
        index_name=INDEX_NAME
    )
    print("Successfully uploaded all embeddings to Pinecone.")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Ensure Pinecone index exists
        create_pinecone_index()
        
        # 2. Load and preprocess data
        document_chunks = load_and_split_documents()
        
        # 3. Embed and store the final chunks
        if document_chunks:
            embed_and_store(document_chunks)
        
    except Exception as e:
        print(f"\nAn error occurred during ingestion: {e}")
        print("Please check your API keys and ensure the 'docs' folder contains files.")
        