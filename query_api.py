import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example transformer overheating reports
documents = [
    "Transformer at West Substation recorded 110°C oil temperature due to cooling fan failure.",
    "High ambient temperature caused transformer at East Substation to exceed safe limits.",
    "Overloaded transformer in South Substation detected internal insulation degradation.",
    "Transformer winding overheating in Central Substation due to excessive harmonic distortion.",
    "Transformer T-789 experienced overheating due to blocked radiator airflow."
]

# Convert documents into embeddings
document_embeddings = np.array([embedding_model.encode(doc) for doc in documents])

# Create FAISS index and store embeddings
vector_dim = document_embeddings.shape[1]  # Get embedding size
faiss_index = faiss.IndexFlatL2(vector_dim)
faiss_index.add(document_embeddings)

# Define a request body for API
class QueryRequest(BaseModel):
    query: str

# Function to retrieve relevant transformer maintenance records
def find_similar_cases(query):
    query_embedding = embedding_model.encode([query])

    # Search for the top 3 most similar reports
    D, I = faiss_index.search(np.array(query_embedding), k=3)

    # Retrieve the matching documents
    retrieved_texts = [documents[idx] for idx in I[0]]

    return retrieved_texts

# Define API endpoint
@app.post("/query/")
def query_api(request: QueryRequest):
    results = find_similar_cases(request.query)
    return {"retrieved_documents": results}

# ✅ Auto-detect the correct port (8000 for local, 10000 for Render)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 for local testing
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
