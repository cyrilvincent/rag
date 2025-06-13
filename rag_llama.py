import os
from llama_index.core import Document, GPTVectorStoreIndex

os.environ["OPENAI_API_KEY"] = ""


files = ["davinci.txt"]
documents = []
for file in files:
    with open(file) as f:
        documents.append(Document(text=f.read()))

index = GPTVectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)
query = "How old was Leonardo when he died?"
response = query_engine.query(query)
print(response)
retriever = index.as_retriever()
nodes = retriever.retrieve("How old was Leonardo when he died?")
print(nodes)



