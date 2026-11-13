from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import os


with open("data/secrets/key.secret") as f:
    os.environ["OPENAI_API_KEY"] = f.read()

documents = SimpleDirectoryReader("data/books").load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
index.storage_context.persist("data/storage")

storage_context = StorageContext.from_defaults(persist_dir="data/storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
