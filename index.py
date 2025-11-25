import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Cargar documentos PDF
docs = []

# Carpeta donde están tus PDFs
folder_path = "documentos/"

for file_name in os.listdir(folder_path):
    if file_name.endswith(".pdf"):
        loader = PyMuPDFLoader(os.path.join(folder_path, file_name))
        loaded_docs = loader.load()        
        docs += loaded_docs



# Dividir en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,       # tamaño del fragmento
    chunk_overlap=150,     # solapamiento entre fragmentos
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)

print(f"Se crearon {len(chunks)} fragmentos de texto.")

# Embeddings (usando la clase oficial)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2') # modelo rápido no muy preciso beta


# Usamos la clase Chroma para guardar los embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db_demo"  # Carpeta donde se guarda la BD
)

print("Base de conocimiento creada y guardada en 'db_demo/' ✅")