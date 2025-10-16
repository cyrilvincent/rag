import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import langchain
# pip install langchain langchain-community langchain-openai faiss-cpu tiktoken

print(langchain.__version__)

with open("data/secrets/key.secret") as f:
    os.environ["OPENAI_API_KEY"] = f.read()

# Chargement des documents
loader = TextLoader("data/books/cyril.txt")
documents = loader.load()

# Découpage des documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Création du vecteur store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Moteur de requête
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

# Requête
query = "Quand est né Cyril et qui sont ses enfants ?"
response = qa_chain.invoke(query)
print(response)

# Accès aux documents similaires
similar_docs = retriever.invoke(query)
for doc in similar_docs:
    print(doc.page_content)
