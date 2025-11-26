import os
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # <-- ADD THIS
# --- Pinecone (v3) ---
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# --- Configuration ---
INDEX_NAME = "slack-kb-index"
DOCS_PATH = "docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # <-- NEW

# --- Initialize Pinecone ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# --- Initialize Embeddings ---
# Using HuggingFace sentence-transformers model for embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- 1. Create Pinecone Index ---
def create_pinecone_index():
    print(f"Checking for existing index: {INDEX_NAME}...")

    # `pc.list_indexes()` returns a list of index names
    if INDEX_NAME not in pc.list_indexes():
        print(f"Index '{INDEX_NAME}' not found. Creating a new one...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # Dimension for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        print("Index created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


# --- 2. Load and Split Documents ---
def load_and_split_documents():
    print(f"Loading documents from '{DOCS_PATH}'...")

    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.*",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    return texts


# --- 3. Embed and Store ---
def embed_and_store(texts):
    print("Embedding and uploading to Pinecone...")

    index = pc.Index(INDEX_NAME)
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    print("Successfully uploaded all embeddings to Pinecone.")


# --- Main ---
if __name__ == "__main__":
    try:
        create_pinecone_index()

        texts = load_and_split_documents()

        if texts:
            embed_and_store(texts)

    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Check API keys and ensure the 'docs' folder contains files.")
