from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

# load the embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")      # Has to be the same embedding model used to create the vector store, otherwise the dimensions won't match and you'll get an error.

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Search for relevant documents
query = "In what year did Tesla begin production of the Roadster?"
# query = "Which island does Spacex lease for its launches in the Pacific?"

# retriever = db.as_retriever(search_kwargs={"k": 3})     # Return the top 3 most similar documents, regardless of their similarity score. This is the default behavior if you don't specify a search type or search kwargs.
retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",       # Only return documents with a similarity score above a certain threshold, which you can specify in the search kwargs. This is useful if you want to ensure that the retrieved documents are sufficiently relevant to the query.
#     search_kwargs={                 # The search kwargs you can specify will depend on the search type you choose. For the "similarity_score_threshold" search type, you can specify the following search kwargs:
#         "k": 5,                     # Return the top 5 most similar documents that meet the similarity score threshold. This is optional, but it can be useful if you want to limit the number of results returned, especially if you have a large vector store and a low similarity score threshold, which could potentially return a large number of documents.
#         "score_threshold": 0.3,     # Only return documents with a similarity score >= 0.3. Score threshold = 1 means the retrieved documents must be identical to the query, while a score threshold = 0 means all documents in the vector store will be returned, regardless of their similarity to the query. The optimal score threshold will depend on your specific use case and the quality of your vector store, so you may need to experiment with different values to find the one that works best for you.
#     }
# )

relevant_docs = retriever.invoke(query)         # Retrieve relevant documents based on the query and the specified search type and search kwargs. The retriever will return a list of documents that are relevant to the query, based on the similarity scores calculated between the query and the documents in the vector store, and filtered according to the search type and search kwargs you specified.

print(f"User query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-"*20)

# Synthetic Questions:

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "Which island does SpaceX lease for its launches in the Pacific?"
# 10. "What was the original name of Microsoft before it became Microsoft?"