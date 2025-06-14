import os
from llama_index.core import Document, GPTVectorStoreIndex

with open("data/secrets/key.secret") as f:
    os.environ["OPENAI_API_KEY"] = f.read()

files = [f.path for f in os.scandir("data/books")]

documents = []
for file in files:
    print(f"Loading {file}")
    with open(file, encoding="utf-8") as f:
        documents.append(Document(text=f.read()))

print("Embedding")
index = GPTVectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine(similarity_top_k=3)
# query = "How old was Leonardo when he died?"
# query = "When Leonardo leave Milan ?"
# query = "Dessine moi un mouton"
query = "Quand est né Cyril et qui sont ces enfants ?"
response = query_engine.query(query)
print(response)
retriever = index.as_retriever()
nodes = retriever.retrieve(query)
print(nodes)



