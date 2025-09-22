import os
import httpx
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

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

# --- Configuración de Pinecone y Groq ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Inicialización con validación
try:
    if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        pinecone_index = pc.Index("chatbot-comercio")
        print("✅ Pinecone inicializado correctamente")
    else:
        pinecone_index = None
        print("⚠️ Pinecone no disponible - falta API key")
        
    if GROQ_API_KEY:
        groq_llm = ChatGroq(temperature=0.7, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
        print("✅ Groq inicializado correctamente")
    else:
        groq_llm = None
        print("⚠️ Groq no disponible - falta API key")
        
except Exception as e:
    print(f"❌ Error inicializando servicios: {e}")
    pinecone_index = None
    groq_llm = None

# --- Funciones de RAG ---
def retrieve_from_pinecone(user_message: str):
    """Busca en el índice de Pinecone."""
    if not pinecone_index:
        return ""
    
    try:
        query_results = pinecone_index.query(
            vector=[0.0] * 1536,  # Vector dummy, as we're not creating embeddings
            top_k=3,
            include_metadata=True
        )
        context = ""
        for result in query_results["matches"]:
            if result.get("score", 0) > 0.7:
                context += result['metadata'].get('text', '') + "\n\n"
        return context
    except Exception as e:
        print(f"Error al buscar en Pinecone: {e}")
        return ""

# --- Endpoint de la API ---
@app.get("/")
def home():
    """Endpoint principal para verificar que la API está funcionando."""
    return {
        "message": "¡API del Chatbot de Comercio Internacional en funcionamiento!",
        "status": "running",
        "services": {
            "pinecone": "✅" if pinecone_index else "❌",
            "groq": "✅" if groq_llm else "❌"
        }
    }

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Endpoint con la estrategia RAG para el chatbot.
    """
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Mensaje de usuario no proporcionado."})

        # Paso 1: Recuperar contexto de Pinecone
        rag_context = retrieve_from_pinecone(user_message)
        
        # Paso 2: Crear un prompt con el contexto
        template = """
        Eres un experto en comercio internacional. 
        Utiliza el siguiente contexto para responder a las preguntas de los usuarios. 
        Si no encuentras información relevante en el contexto, simplemente responde que no puedes ayudar con esa pregunta específica.

        Contexto: {context}

        Pregunta: {question}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Paso 3: Crear la cadena de procesamiento
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | groq_llm
            | StrOutputParser()
        )

        # Paso 4: Invocar la cadena para obtener la respuesta
        response = chain.invoke({"context": rag_context, "question": user_message})

        return {"response": response}
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error interno: {str(e)}"})

# Para Vercel, necesitamos exportar la app
handler = app