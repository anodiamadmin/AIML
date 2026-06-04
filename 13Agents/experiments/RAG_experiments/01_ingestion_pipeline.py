# ============================================================================
# Ingestion Pipeline for Company Documents
# This script loads text files from the "docs" directory, splits them into smaller chunks,
# embeds the chunks and stores them in a vector database.
# ============================================================================

# ============================================================================
# Step 0: Import the necessary libraries
# Make sure to install the required libraries before running this script:
# $> pip install langchain langchain_community langchain-text-splitters langchain-openai langchain-chroma python-dotenv 
# ============================================================================
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader    # DirectoryLoader is used to load multiple documents from a directory
from langchain_text_splitters import CharacterTextSplitter      # CharacterTextSplitter is used to split the documents into smaller chunks based on character count
from langchain_openai import OpenAIEmbeddings           # OpenAIEmbeddings is used to create vector embeddings from the document chunks using OpenAI's embedding models
from langchain_chroma import Chroma             # Chroma db is a Vector database that is hosted locally
from dotenv import load_dotenv          # load_dotenv is used to load environment variables from a .env file, which is where you should store your OpenAI API key and other configuration settings.

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
        glob="*.txt",           # only load .txt files, you can modify this to load other file types if needed
        loader_cls=TextLoader,  # use TextLoader to load the content of the files as text documents. You can change this to a different loader if you have other file formats (e.g. PDF, Word, website etc.)
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
        # print(f"  Full Content:\n{doc.page_content}")
        # print(f"  Document Object: {doc}")
        print("-*" * 20)
    
    print("=" * 40)
    
    return documents


# ============================================================================
# Step 2: Function for splitting documents into chunks
# ============================================================================
def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks."""
    print("=" * 40)
    print(f"\n#2# Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")
    
    text_splitter = CharacterTextSplitter(      # use CharacterTextSplitter to split the documents into chunks based on character count. You can change this to a different splitter if you want to split based on sentences, paragraphs, or other criteria.
        chunk_size=chunk_size,                  # the maximum number of characters in each chunk. You can adjust this based on the average length of your documents and the context window size of the language model you plan to use for embeddings and retrieval.
        chunk_overlap=chunk_overlap             # the number of characters to overlap between chunks. This can help maintain context across chunks, but it will also increase the total number of chunks and the size of the vector store. You can adjust this based on your specific use case and the nature of your documents.
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
    
    # Create embeddings:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")    # use OpenAI's embedding model to create vector embeddings from the document chunks. You can choose different embedding models based on your needs and the size of your documents. The "text-embedding-3-small" model is a good starting point for general-purpose embeddings, but you can experiment with other models like "text-embedding-3-large" or "text-embedding-3-mini" depending on your requirements for embedding quality and computational resources.
    
    # Create Chroma vector store
    print(f"\n--- Creating Chroma vector store ---")
    vector_store = Chroma.from_documents(               # create a Chroma vector store from the document chunks and the embedding model. The from_documents method will take care of embedding the chunks and storing them in the vector database. You can adjust the collection_metadata parameter to set different configurations for the vector store, such as the distance metric (e.g. cosine, euclidean, etc.) or other settings that Chroma supports.
        documents=chunks,                               # the list of document chunks to be embedded and stored in the vector database. Each chunk should be a Document object that contains the content and metadata of the chunk.
        embedding=embedding_model,                      # the embedding model to use for creating vector embeddings from the document chunks. This should be an instance of an embedding class that implements the necessary interface for generating embeddings.
        persist_directory=persist_directory,            # the directory where the Chroma vector store will be persisted. This should be a path on your local file system where you want to save the vector database. You can choose any directory you like, but make sure it is writable and has enough space to store the vector data.
        collection_metadata={"hnsw:space": "cosine"}    # optional metadata for the collection, in this case we are specifying that we want to use cosine distance for similarity search. You can adjust this based on your specific use case and the distance metric you prefer for retrieval.
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
    documents = load_documents(docs_path="docs")    # this will load all the .txt files from the "docs" directory and return a list of Document objects. Each Document object contains the content of the file as well as metadata such as the source (file path) and any other relevant information. You can modify the load_documents function to load different types of files (e.g. PDF, Word, website etc.) by using different loaders provided by LangChain or by implementing your own custom loader if needed.
    
    #2. Split the documents into chunks
    chunks = split_documents(documents)             # you can adjust the chunk_size and chunk_overlap parameters in the split_documents function to optimize the chunking based on your specific documents and use case. For example, if your documents are very long, you might want to increase the chunk_size to reduce the total number of chunks and improve retrieval performance. Conversely, if your documents are short or you want to maintain more granular context, you might want to decrease the chunk_size. The chunk_overlap can help maintain context across chunks, but it will also increase the total number of chunks and the size of the vector store, so you should adjust it based on your specific needs and constraints.
    
    #3. Embed the chunks and store in vector database (to be implemented in next steps)
    vector_store = create_vector_store(chunks)      # this will create the vector store and persist it to disk, so you only need to run this once. In future runs, you can load the existing vector store from disk instead of creating it again.


# ============================================================================
# Run the main function
# ============================================================================
if __name__ == "__main__":
    main()