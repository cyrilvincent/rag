from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
import textwrap
import openai

import os

with open("data/secrets/key.secret") as f:
    os.environ["OPENAI_API_KEY"] = f.read()

import psycopg2
connection_string = "postgresql://postgres:sa@localhost:5433"
conn = psycopg2.connect(connection_string)
conn.autocommit = True

documents = SimpleDirectoryReader("data/books").load_data()
from sqlalchemy import make_url

url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database="vector",
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="books",
    embed_dim=1536,  # openai embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)
query_engine = index.as_query_engine()

response = query_engine.query("A quel age est mort De Vinci?")
print(textwrap.fill(str(response), 100))
