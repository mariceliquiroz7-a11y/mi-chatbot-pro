import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Cargar variables de entorno
load_dotenv()

# --- Configuración de Pinecone y Cohere ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# --- Funciones de RAG y Búsqueda Web ---
def create_embeddings_cohere(texts):
    """Crea embeddings para el texto usando la API de Cohere."""
    if not COHERE_API_KEY:
        print("Error: COHERE_API_KEY no está configurada.")
        return None
        
    try:
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json",
        }
        json_data = {
            "texts": texts,
            "model": "embed-multilingual-v3.0",
            "input_type": "search_document"
        }
        
        response = requests.post(
            "https://api.cohere.ai/v1/embed",
            headers=headers,
            json=json_data
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except requests.exceptions.RequestException as e:
        print(f"Error al crear embeddings con Cohere: {e}")
        return None

def main():
    print("🚀 Iniciando indexación para Pinecone...")
    
    # 1. Cargar documentos PDF y URLs
    docs_path = Path("./docs")
    documents = []
    
    if docs_path.exists():
        pdf_files = list(docs_path.glob("*.pdf"))
        if pdf_files:
            print(f"📄 Procesando {len(pdf_files)} archivos PDF...")
            pdf_documents = SimpleDirectoryReader(input_dir="./docs").load_data()
            documents.extend(pdf_documents)
            print(f"✅ PDFs cargados: {len(pdf_documents)} documentos")
        
    print("🌐 Cargando recursos web...")
    urls = [
        "https://es.wikipedia.org/wiki/Comercio_internacional",
        "https://es.wikipedia.org/wiki/Organizaci%C3%B3n_Mundial_del_Comercio",
        "https://es.wikipedia.org/wiki/Arancel",
        "https://es.wikipedia.org/wiki/Incoterms",
    ]
    try:
        web_reader = SimpleWebPageReader(html_to_text=True)
        url_documents = web_reader.load_data(urls)
        documents.extend(url_documents)
        print(f"✅ Páginas web cargadas: {len(url_documents)} recursos")
    except Exception as e:
        print(f"⚠️ Error al cargar URLs: {e}")

    if not documents:
        print("❌ No se encontraron documentos para indexar.")
        return
        
    print(f"📚 Total de documentos: {len(documents)}")

    # 2. Conectar a Pinecone
    try:
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
            print("❌ Falta una clave API o entorno de Pinecone.")
            return
            
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        index_name = "chatbot-comercio"
        
        if index_name not in pc.list_indexes().names:
            print("🔧 Creando índice en Pinecone...")
            pc.create_index(
                name=index_name,
                dimension=1024,  # La dimensión del modelo Cohere
                metric="cosine",
            )
        else:
            print(f"✅ Conectado al índice existente: '{index_name}'")

        pinecone_index = pc.Index(index_name)

    except Exception as e:
        print(f"❌ Error al conectar con Pinecone: {e}")
        return

    # 3. Crear embeddings e indexar
    print("🧠 Creando embeddings e indexando documentos...")
    nodes = []
    for doc in documents:
        text = doc.get_content()
        embedding = create_embeddings_cohere([text])
        if embedding:
            node = {"id": doc.doc_id, "values": embedding[0], "metadata": {"text": text}}
            nodes.append(node)

    if nodes:
        pinecone_index.upsert(vectors=nodes)
        print("🎉 ¡Indexación completada!")
        print(f"📊 Documentos procesados: {len(documents)}")
        print("🚀 Tu índice de Pinecone está listo para el chatbot.")
    else:
        print("⚠️ No se pudieron crear los embeddings. Indexación fallida.")

if __name__ == "__main__":
    main()