import os
from llama_index.core import Document, GPTVectorStoreIndex, Settings
from llama_index.core.schema import NodeRelationship
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
import numpy as np

# pip install llama-index-embeddings-mistralai
# pip install llama-index-llms-mistralai

with open("data/secrets/mistral.secret") as f:
    api_key = f.read()

model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)

Settings.embed_model = embed_model
files = [f.path for f in os.scandir("data/bale3")]

def extract_nodes(vector_store):
    node_dict = {}
    for node_id, metadata in vector_store._data.metadata_dict.items():
        node_dict[node_id] = metadata
    return node_dict


documents = []
for file in files:
    print(f"Loading {file}")
    with open(file, encoding="utf-8") as f:
        documents.append(Document(text=file + "\n" + f.read()))

llm=MistralAI(api_key=api_key)
print("Embedding")
index = GPTVectorStoreIndex.from_documents(documents, show_progress=True, llm=llm)

vector_store = index.storage_context.vector_store
vector_store_dict = vector_store.to_dict()
keys = extract_nodes(vector_store)


query_engine = index.as_query_engine(similarity_top_k=3, llm=llm)
query = "Qu'est ce qu'un test de résistance"
response = query_engine.query(query)
print(response)
retriever = index.as_retriever()
nodes = retriever.retrieve(query)
print(nodes)
for node in nodes:
    print(f"Node {node.id_}")
    print(f"Score {node.score}")
    document_id = keys[node.id_]["document_id"]
    print(f"Document id {document_id}")
    document = [d for d in documents if d.id_ == document_id][0]
    print(f"From document {document.text.split("\n")[0]}")
    print(node.text[:255])
    print()



