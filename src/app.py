from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from src.constants import *
from src.logger import logging
from src.exception import *
from database.kdb_init import table
from llama_index.core import StorageContext
import nest_asyncio
nest_asyncio.apply()
# Import your existing classes and functions
from src.main import DocumentProcessor, VectorStoreIndex, CohereRerank,  KDBAIVectorStore, query_engine, base_nodes, objects

# Create an instance of FastAPI
app = FastAPI()

# Create an instance of your DocumentProcessor
processor = DocumentProcessor()

# Create an instance of your CohereRerank
cohere_rerank = CohereRerank(top_n=10)

# Create an instance of VectorStoreIndex
# Note: You may need to adjust the arguments based on your existing code
vector_store = KDBAIVectorStore(table)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
recursive_index = VectorStoreIndex(nodes=base_nodes + objects, storage_context=storage_context)

# Define a request model for your query
class QueryRequest(BaseModel):
    query: str

# Define a response model for your query
class QueryResponse(BaseModel):
    response: str


logging.info("Starting FastAPI application")

# Define a FastAPI endpoint for your query
@app.post("/query/", response_model=QueryResponse)
async def query_endpoint(query_request: QueryRequest):
    logging.info(f"Received query: {query_request.query} from {request.client.host}")
    try:
        response = query_engine.query(query_request.query)
        logging.info(f"Generated response: {response}")
        return {"response": str(response)}
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    logging.info("Running FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
