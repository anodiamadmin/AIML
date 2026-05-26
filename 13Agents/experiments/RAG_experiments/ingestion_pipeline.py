# ============================================================================
# Ingestion Pipeline for Company Documents
# This script loads text files from the "docs" directory, splits them into smaller chunks,
# embeds the chunks and stores them in a vector database.
# ============================================================================

# ============================================================================
# Step 0: Import the necessary libraries
# ============================================================================
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Step 1: Function for loading documents
# ============================================================================

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""
    print("=" * 40)
    print(f"\n#1# Loading documents from {docs_path}...")
    
    # check if the docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist. Please create it and add your company files.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True
        }
    )
    
    documents = loader.load()
    
    if(len(documents) == 0):
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]):     # print the first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content Length: {len(doc.page_content)} characters")
        print(f"  Content Preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")
    
    print("=" * 40)
    
    return documents


# ============================================================================
# Step 2: Function for splitting documents into chunks
# ============================================================================
def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks."""
    print("=" * 40)
    print(f"\n#2# Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        print(f"\nCreated {len(chunks)} chunks from {len(documents)} documents.")
        for i, chunk in enumerate(chunks[:5]):     # print the first 5 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"  Content Length: {len(chunk.page_content)} characters")
            print(f"  Metadata: {chunk.metadata}")
            print(f"  Content:\n{chunk.page_content}")
            print("-" * 20)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks.")
        
    print("=" * 40)
    
    return chunks


# ============================================================================
# Step 3: Function for embedding chunks and storing in vector database
# ============================================================================
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist Chroma DB vector store from the document chunks."""
    print("=" * 40)
    print(f"\n#3# Creating embeddings and storing in ChromaDB...")
    
    # Create embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create Chroma vector store
    print(f"\n--- Creating Chroma vector store ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to '{persist_directory}'.")
    print("=" * 40)
    
    return vector_store


# ============================================================================
# Main function to run the ingestion pipeline
# ============================================================================
def main():
    print("Starting ingestion pipeline...")
    
    #1. Load the files
    documents = load_documents(docs_path="docs")
    #2. Split the documents into chunks
    chunks = split_documents(documents)
    #3. Embed the chunks and store in vector database (to be implemented in next steps)
    vector_store = create_vector_store(chunks)

# ============================================================================
# Run the main function
# ============================================================================
if __name__ == "__main__":
    main()