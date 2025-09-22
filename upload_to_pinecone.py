import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone

# 1. Cargar tus documentos desde la carpeta 'docs'
print("Cargando documentos...")
documents = SimpleDirectoryReader("docs").load_data()
print(f"Se encontraron {len(documents)} documentos.")

# 2. Configurar la conexión a Pinecone
print("Conectando a Pinecone...")
api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT")
index_name = "chatbot-comercio"

if not api_key or not environment:
    raise ValueError("Las variables de entorno PINECONE_API_KEY y PINECONE_ENVIRONMENT no están configuradas.")

# 3. Inicializar Pinecone y el vector store
pc = Pinecone(api_key=api_key, environment=environment)
pinecone_index = pc.Index(index_name)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Usar el mismo modelo de embeddings que elegiste en Pinecone
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")

# 5. Crear el índice y subir los documentos a Pinecone en lotes
print("Subiendo documentos a Pinecone en lotes...")

# Puedes ajustar el tamaño del lote aquí
batch_size = 5

for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i + batch_size]
    print(f"Procesando lote {i // batch_size + 1}/{len(documents) // batch_size + 1}...")

    index = VectorStoreIndex.from_documents(
        batch_docs,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print(f"Lote {i // batch_size + 1} subido con éxito.")

print("¡Todos los documentos han sido subidos con éxito a Pinecone!")