from src.constants import *
from src.logger import logging
from src.exception import *
import kdbai_client as kdbai


logging.info("Connecting to KDB.AI")
#connect to KDB.AI
session = kdbai.Session(api_key=KDB_AI_API_KEY, endpoint=KDB_ENDPOINT)
logging.info("Successfully connected to KDB.AI")

# The schema contains two metadata columns (document_id, text) and one embeddings column
# Index type, search metric (Euclidean distance), and dimensions are specified in the embedding column
schema = dict(
    columns=[
        dict(name="document_id", pytype="bytes"),
        dict(name="text", pytype="bytes"),
        dict(
            name="embedding",
            vectorIndex=dict(type="flat", metric="L2", dims=1536),
        ),
    ]
)

KDBAI_TABLE_NAME = "LlamaParse_Table"

# First ensure the table does not already exist
if KDBAI_TABLE_NAME in session.list():
    logging.info(f"Table {KDBAI_TABLE_NAME} already exists. Dropping the table.")
    session.table(KDBAI_TABLE_NAME).drop()
    logging.info(f"Table {KDBAI_TABLE_NAME} dropped.")


logging.info(f"Creating table {KDBAI_TABLE_NAME}")
#Create the table
table = session.create_table(KDBAI_TABLE_NAME, schema)
logging.info(f"Table {KDBAI_TABLE_NAME} created successfully")