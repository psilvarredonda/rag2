import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

# Cargar variables de entorno
load_dotenv()
# api_key = os.getenv("GENAI_API_KEY")
api_key = st.secrets["GENAI_API_KEY"]

# Crear cliente de Google GenAI
client = genai.Client(api_key=api_key)

# Crear embeddings y vectorstore (IMPORTANTE: DEBES TENER TU CHROMA CREADO)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')

# Si YA tienes una base Chroma creada
vectorstore = Chroma(
    persist_directory="db_demo",  # cambia si tu carpeta se llama diferente
    embedding_function=embeddings
)

# Interfaz Streamlit
st.title("Título")
st.write("TEXTO")

query = st.text_input("Ingresa tu pregunta:")

if st.button("Buscar respuesta"):
    if query.strip() == "":
        st.warning("Por favor escribe una pregunta.")
    else:
        # Buscar documentos relevantes
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([d.page_content[:1000] for d in docs])

        prompt = f"""
Usa el siguiente contexto para responder la pregunta de manera clara y completa.

CONTEXTO:
{context}

PREGUNTA:
{query}

RESPUESTA:
"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer = response.text
        except Exception as e:
            answer = f"⚠️ Error al generar la respuesta: {e}"

        st.subheader("💬 Respuesta")
        st.write(answer)

        with st.expander("📄 Contexto completo usado:"):

            st.text(context)
