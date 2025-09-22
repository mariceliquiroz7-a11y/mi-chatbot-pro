import os
import httpx
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from duckduckgo_search import DDGS
import cohere

# Cargar variables de entorno
# load_dotenv() - COMENTADO: Vercel maneja las env vars automáticamente

# --- Configuración de la API con FastAPI ---
app = FastAPI(title="Chatbot de Comercio Internacional")

# Permitir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuración de Pinecone y Cohere ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Inicialización con validación
try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index("chatbot-comercio")
        print("✅ Pinecone inicializado correctamente")
    else:
        pinecone_index = None
        print("⚠️ Pinecone no disponible - falta API key")
        
    if COHERE_API_KEY:
        cohere_client = cohere.Client(api_key=COHERE_API_KEY)
        print("✅ Cohere inicializado correctamente")
    else:
        cohere_client = None
        print("⚠️ Cohere no disponible - falta API key")
        
except Exception as e:
    print(f"❌ Error inicializando servicios: {e}")
    pinecone_index = None
    cohere_client = None

# --- Funciones de RAG y Búsqueda Web ---
def create_embeddings(text):
    """Crea embeddings para el texto usando la API de Cohere (ligero)."""
    if not cohere_client:
        return None
        
    try:
        response = cohere_client.embed(
            texts=[text],
            model="embed-multilingual-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"Error al crear embeddings: {e}")
        return None

async def retrieve_from_pinecone(user_message: str):
    """Busca en el índice de Pinecone usando la API de Cohere para embeddings."""
    if not pinecone_index or not cohere_client:
        print("⚠️ Pinecone o Cohere no disponible")
        return ""
        
    try:
        query_embedding = create_embeddings(user_message)
        if not query_embedding:
            return ""
            
        query_results = pinecone_index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        context = ""
        for result in query_results["matches"]:
            if result.get("score", 0) > 0.7:  # Solo resultados relevantes
                context += result['metadata'].get('text', '') + "\n\n"
        return context
    except Exception as e:
        print(f"Error al buscar en Pinecone: {e}")
        return ""

async def web_search_with_duckduckgo(user_message: str):
    """Realiza una búsqueda web priorizando fuentes oficiales."""
    try:
        # Búsqueda específica en sitios gubernamentales peruanos
        official_query = f"{user_message} site:gob.pe OR site:sunat.gob.pe OR site:mincetur.gob.pe"
        
        # Timeout más corto para evitar problemas en serverless
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=official_query, max_results=2))
        
        # Si no hay resultados oficiales, búsqueda general
        if not results:
            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=f"{user_message} comercio internacional Perú", max_results=3))
        
        web_context = ""
        for r in results:
            web_context += f"[FUENTE WEB] {r['title']}: {r['body']}\n\n"
        
        return web_context
    except Exception as e:
        print(f"Error en la búsqueda web: {e}")
        return ""

# --- Endpoints de la API ---
@app.get("/")
def home():
    """Endpoint principal para verificar que la API está funcionando."""
    return {
        "message": "¡API del Chatbot de Comercio Internacional en funcionamiento!",
        "status": "running",
        "services": {
            "pinecone": "✅" if pinecone_index else "❌",
            "cohere": "✅" if cohere_client else "❌",
            "groq": "✅" if GROQ_API_KEY else "❌",
            "openrouter": "✅" if OPENROUTER_API_KEY else "❌"
        }
    }

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Endpoint con la estrategia híbrida para el chatbot.
    """
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Mensaje de usuario no proporcionado."})

        # PASO 1: Buscar en conocimiento especializado (RAG) - solo si está disponible
        rag_result = ""
        if pinecone_index and cohere_client:
            print("🔍 Buscando en documentos especializados...")
            rag_result = await retrieve_from_pinecone(user_message)
        
        # PASO 2: Si RAG es insuficiente, buscar información actualizada
        web_search_result = ""
        if len(rag_result.strip()) < 100:
            print("🌐 Ampliando búsqueda con información web actualizada...")
            web_search_result = await web_search_with_duckduckgo(user_message)

        # PASO 3: Construir contexto híbrido para el LLM
        if rag_result or web_search_result:
            combined_context = f"=== CONOCIMIENTO ESPECIALIZADO ===\n{rag_result}\n\n=== INFORMACIÓN ACTUALIZADA ===\n{web_search_result}"
            final_prompt = f"Basado en el siguiente contexto:\n\n{combined_context}\n\nResponde a la pregunta del usuario de manera detallada y precisa: {user_message}"
        else:
            final_prompt = f"Responde como experto en comercio internacional: {user_message}"

        # PASO 4: Decidir qué LLM usar según complejidad
        message_words = user_message.lower().split()
        is_complex = any(keyword in user_message.lower() for keyword in 
                             ['explica', 'analiza', 'compara', 'diferencia', 'ventajas', 'desventajas', 'cómo funciona', 'proceso',
                              'explain', 'analyze', 'compare', 'difference', 'advantages', 'disadvantages', 'how does', 'process'])
        
        use_groq = is_complex or len(message_words) > 8

        # PASO 5: Generar respuesta con fallback automático
        response = None
        
        if use_groq and GROQ_API_KEY:
            print(f"🟢 Usando Groq para pregunta compleja...")
            response = await call_groq_api(final_prompt)
            
        if not response and OPENROUTER_API_KEY:
            print(f"🟡 Usando OpenRouter...")
            response = await call_openrouter_api(final_prompt)
            
        if not response and GROQ_API_KEY and not use_groq:
            print("⚠️ Fallback a Groq...")
            response = await call_groq_api(final_prompt)

        if not response:
            return JSONResponse(status_code=503, content={
                "error": "Servicio no disponible", 
                "details": "No hay APIs de LLM disponibles. Verifica las claves API."
            })
            
        return {"response": response}
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error interno: {str(e)}"})

# --- FUNCIONES CORREGIDAS DE LLAMADAS A APIs ---
async def call_groq_api(user_message: str) -> str:
    """Llamada a Groq API con manejo de errores."""
    if not GROQ_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:  # Timeout reducido para serverless
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Eres un asistente experto en comercio internacional. Proporciona respuestas detalladas y precisas sobre aranceles, importaciones, exportaciones, documentación comercial, y regulaciones internacionales."
                        },
                        {"role": "user", "content": user_message}
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"✅ Groq response length: {len(content) if content else 0}")
                return content
            else:
                print(f"❌ Error Groq: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error llamando Groq: {e}")
        return None

async def call_openrouter_api(user_message: str) -> str:
    """Llamada a OpenRouter API con manejo de errores."""
    if not OPENROUTER_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:  # Timeout reducido para serverless
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://chatbot-de-mari-bo2a6ihn3-maris-projects-da1a1ada.vercel.app",
                    "X-Title": "Chatbot Comercio Internacional"
                },
                json={
                    "model": "mistralai/mistral-7b-instruct:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Eres un asistente de comercio internacional. Responde de forma clara y concisa."
                        },
                        {"role": "user", "content": user_message}
                    ],
                    "max_tokens": 800,
                    "temperature": 0.6
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"✅ OpenRouter response length: {len(content) if content else 0}")
                return content
            else:
                print(f"❌ Error OpenRouter: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error llamando OpenRouter: {e}")
        return None

# Para Vercel, necesitamos exportar la app
handler = app