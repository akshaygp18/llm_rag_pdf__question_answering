from llama_parse import LlamaParse
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.kdbai import KDBAIVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
import logging
from typing import Any, List, Callable, Optional

import pandas as pd

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.kdbai.utils import default_sparse_encoder
import kdbai_client as kdbai
from database.kdb_init import *
from src.constants import *
from src.logger import logging
from src.exception import *
from database.kdb_init import table




class DocumentProcessor:
    def __init__(self, generation_model=GPT_MODEL, embedding_model="text-embedding-3-small", llama_api_key=LLAMA_CLOUD_API):
        logging.info("Initializing DocumentProcessor")
        self.llm = OpenAI(model=generation_model)
        self.embed_model = OpenAIEmbedding(model=embedding_model)
        self.llama_api_key = llama_api_key

    def load_documents(self, pdf_file_name):
        logging.info(f"Loading documents from {pdf_file_name}")
        parsing_instructions = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n

            Answer:
            """
        documents = LlamaParse(api_key=self.llama_api_key, result_type="markdown", parsing_instructions=parsing_instructions).load_data(pdf_file_name)
        logging.info("Documents loaded successfully")
        return documents

    def parse_documents(self, documents):
        logging.info("Parsing documents")
        node_parser = MarkdownElementNodeParser(llm=self.llm, num_workers=8).from_defaults()
        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        logging.info("Documents parsed successfully")
        return base_nodes, objects

    def convert_metadata_col(self, column, value):
        try:
            logging.info(f"Converting metadata column {column['name']}")
            if column["pytype"] == "str":
                return str(value)
            elif column["pytype"] == "bytes":
                return value.encode("utf-8")
            elif column["pytype"] == "datetime64[ns]":
                return pd.to_datetime(value)
            elif column["pytype"] == "timedelta64[ns]":
                return pd.to_timedelta(value)
            return value.astype(column["pytype"])
        except Exception as e:
            logging.error(
                f"Failed to convert column {column['name']} to type {column['pytype']}: {e}"
            )

# Usage
logging.info("Starting DocumentProcessor")
processor = DocumentProcessor()
pdf_file_name = 'files/Transformer.pdf'
documents = processor.load_documents(pdf_file_name)
base_nodes, objects = processor.parse_documents(documents)

DEFAULT_COLUMN_NAMES = ["document_id", "text", "embedding"]

DEFAULT_BATCH_SIZE = 100

class KDBAIVectorStore(BasePydanticVectorStore):
    """The KDBAI Vector Store.

    In this vector store we store the text, its embedding and
    its metadata in a KDBAI vector store table. This implementation
    allows the use of an already existing table.

    Args:
        table kdbai.Table: The KDB.AI table to use as storage.
        batch (int, optional): batch size to insert data.
            Default is 100.

    Returns:
        KDBAIVectorStore: Vectorstore that supports add and query.
    """

    stores_text: bool = True
    flat_metadata: bool = True

    hybrid_search: bool = False
    batch_size: int

    _table: Any = PrivateAttr()
    _sparse_encoder: Optional[Callable] = PrivateAttr()

    def __init__(
        self,
        table: Any = None,
        hybrid_search: bool = False,
        sparse_encoder: Optional[Callable] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        try:
            import kdbai_client as kdbai

            logging.info("KDBAI client version: " + kdbai.__version__)

        except ImportError:
            raise ValueError(
                "Could not import kdbai_client package."
                "Please add it to the dependencies."
            )

        if table is None:
            logging.error("Must provide an existing KDB.AI table.")
            raise ValueError("Must provide an existing KDB.AI table.")
        else:
            self._table = table

        if hybrid_search:
            if sparse_encoder is None:
                self._sparse_encoder = default_sparse_encoder
            else:
                self._sparse_encoder = sparse_encoder

        super().__init__(batch_size=batch_size, hybrid_search=hybrid_search)

    @property
    def client(self) -> Any:
        """Return KDB.AI client."""
        return self._table

    @classmethod
    def class_name(cls) -> str:
        return "KDBAIVectorStore"

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to the KDBAI Vector Store.

        Args:
            nodes (List[BaseNode]): List of nodes to be added.

        Returns:
            List[str]: List of document IDs that were added.
        """
        df = pd.DataFrame()
        docs = []
        schema = self._table.schema()["columns"]
        if self.hybrid_search:
            schema = [item for item in schema if item["name"] != "sparseVectors"]

        try:
            logging.info("Adding nodes to the KDBAI Vector Store")
            for node in nodes:
                doc = {
                    "document_id": node.node_id,
                    "text": node.text.encode("utf-8"),
                    "embedding": node.embedding,
                }

                if self.hybrid_search:
                    doc["sparseVectors"] = self._sparse_encoder([node.get_content()])

                # handle extra columns
                if len(schema) > len(DEFAULT_COLUMN_NAMES):
                    for column in schema[len(DEFAULT_COLUMN_NAMES) :]:
                        try:
                            doc[column["name"]] = convert_metadata_col(
                                column, node.metadata[column["name"]]
                            )
                        except Exception as e:
                            logging.error(
                                f"Error writing column {column['name']} as type {column['pytype']}: {e}."
                            )

                docs.append(doc)

            df = pd.DataFrame(docs)
            for i in range((len(df) - 1) // self.batch_size + 1):
                batch = df.iloc[i * self.batch_size : (i + 1) * self.batch_size]
                try:
                    self._table.insert(batch, warn=False)
                    logging.info(f"inserted batch {i}")
                except Exception as e:
                    logging.exception(
                        f"Failed to insert batch {i} of documents into the datastore: {e}"
                    )
            logging.info("Nodes added successfully")
            return df["document_id"].tolist()

        except Exception as e:
            logging.error(f"Error preparing data for KDB.AI: {e}.")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        logging.info("Querying the KDBAI Vector Store")
        if query.filters is None:
            filter = []
        else:
            filter = query.filters

        if self.hybrid_search:
            alpha = query.alpha if query.alpha is not None else 0.5
            sparse_vectors = self._sparse_encoder([query.query_str])
            results = self._table.hybrid_search(
                dense_vectors=[query.query_embedding],
                sparse_vectors=sparse_vectors,
                n=query.similarity_top_k,
                filter=filter,
                alpha=alpha,
            )[0]
        else:
            results = self._table.search(
                vectors=[query.query_embedding], n=query.similarity_top_k, filter=filter
            )[0]

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []

        for result in results.to_dict(orient="records"):
            metadata = {x: result[x] for x in result if x not in DEFAULT_COLUMN_NAMES}
            node = TextNode(
                text=result["text"], id_=result["document_id"], metadata=metadata
            )
            top_k_ids.append(result["document_id"])
            top_k_nodes.append(node)
            top_k_scores.append(result["__nn_distance"])
        
        logging.info("Query completed successfully")
        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    def delete(self, **delete_kwargs: Any) -> None:
        raise Exception("Not implemented.")

logging.info("Initializing KDBAIVectorStore")
vector_store = KDBAIVectorStore(table)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Create the index, inserts base_nodes and objects into KDB.AI
recursive_index = VectorStoreIndex(
    nodes= base_nodes + objects, storage_context=storage_context
)

# Query KDB.AI to ensure the nodes were inserted
table.query()


### Define reranker
cohere_rerank = CohereRerank(top_n=10)

### Create the query_engine to execute RAG pipeline using LlamaIndex, KDB.AI, and Cohere reranker
query_engine = recursive_index.as_query_engine(similarity_top_k=15, node_postprocessors=[cohere_rerank])

query_1 = "what is the use of transformer"

response = query_engine.query(query_1)
logging.info(f"Generated response: {str(response)}")
print(str(response))