import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import SimpleWebPageReader
import chromadb
from pathlib import Path

def main():
    print("🚀 Iniciando indexación híbrida para comercio internacional...")
    
    try:
        # 1. Configurar modelo de embeddings local (GRATIS, SIN LÍMITES)
        print("📦 Configurando embeddings locales...")
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./embedding_cache"
        )
        Settings.embed_model = embed_model
        print("✅ Modelo de embeddings configurado (local, sin límites)")
        
        # 2. Cargar documentos PDF
        docs_path = Path("./docs")
        if not docs_path.exists():
            docs_path.mkdir()
        
        pdf_files = list(docs_path.glob("*.pdf"))
        documents = []
        
        if pdf_files:
            print(f"📄 Procesando {len(pdf_files)} archivos PDF...")
            pdf_documents = SimpleDirectoryReader(input_dir="./docs").load_data()
            documents.extend(pdf_documents)
            print(f"✅ PDFs cargados: {len(pdf_documents)} documentos")
        
        # 3. URLs de comercio internacional y web scraping
        print("🌐 Cargando recursos web de comercio internacional...")
        urls = [
            # Comercio Internacional Oficial
            "https://es.wikipedia.org/wiki/Comercio_internacional",
            "https://es.wikipedia.org/wiki/Organizaci%C3%B3n_Mundial_del_Comercio",
            "https://es.wikipedia.org/wiki/Arancel",
            "https://es.wikipedia.org/wiki/Incoterms",
            
            # Web Scraping
            "https://es.wikipedia.org/wiki/Web_scraping",
            "https://es.wikipedia.org/wiki/Extracci%C3%B3n_de_datos",
            
            # Logística y Aduanas
            "https://es.wikipedia.org/wiki/Log%C3%ADstica",
            "https://es.wikipedia.org/wiki/Aduana"
        ]
        
        try:
            web_reader = SimpleWebPageReader(html_to_text=True)
            url_documents = web_reader.load_data(urls)
            documents.extend(url_documents)
            print(f"✅ Páginas web cargadas: {len(url_documents)} recursos")
        except Exception as e:
            print(f"⚠️ Error con algunas URLs: {e}")
            print("Continuando con documentos disponibles...")
        
        if not documents:
            print("❌ No se encontraron documentos para indexar")
            return
        
        print(f"📚 Total de documentos: {len(documents)}")
        
        # 4. Configurar base de datos vectorial
        print("🔧 Configurando ChromaDB...")
        db = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            db.delete_collection("comercio_internacional_hybrid")
        except:
            pass
        
        chroma_collection = db.create_collection("comercio_internacional_hybrid")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 5. Crear índice vectorial
        print("🧠 Creando índice vectorial híbrido...")
        print("⏱️ Primera ejecución: descarga modelo (~90MB)")
        
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            show_progress=True
        )
        
        print("\n🎉 ¡Indexación completada!")
        print("📊 Resumen:")
        print(f"   - Documentos procesados: {len(documents)}")
        print(f"   - Embeddings: all-MiniLM-L6-v2 (local)")
        print(f"   - Base de datos: ./chroma_db")
        print(f"   - Colección: comercio_internacional_hybrid")
        print("\n🚀 Listo para el sistema híbrido de respuestas!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Posibles causas:")
        print("  - Conexión a internet (descarga del modelo)")
        print("  - Espacio en disco insuficiente")
        print("  - PDFs corruptos")

if __name__ == "__main__":
    main()