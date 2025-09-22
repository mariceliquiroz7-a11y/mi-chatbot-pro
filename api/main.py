import os
from flask import Flask, request, jsonify
from pinecone import Pinecone
from groq import Groq

# --- Configuración de la aplicación con Flask ---
app = Flask(__name__)

# --- Configuración de Pinecone y Groq ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index("chatbot-comercio")
        print("✅ Pinecone inicializado correctamente")
    else:
        pinecone_index = None
        print("⚠️ Pinecone no disponible - falta API key")
        
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq inicializado correctamente")
    else:
        groq_client = None
        print("⚠️ Groq no disponible - falta API key")
        
except Exception as e:
    print(f"❌ Error inicializando servicios: {e}")
    pinecone_index = None
    groq_client = None

# --- Funciones de RAG ---
def retrieve_from_pinecone(user_message: str):
    """Busca en el índice de Pinecone."""
    if not pinecone_index:
        return ""
    
    try:
        query_results = pinecone_index.query(
            vector=[0.0] * 1024,
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

def generate_response(prompt: str):
    """Genera una respuesta usando la API de Groq."""
    if not groq_client:
        return "Lo siento, el servicio de IA no está disponible."
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error al generar respuesta con Groq: {e}")
        return "Lo siento, ha ocurrido un error al generar la respuesta."

# --- Endpoints de la API ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "¡API del Chatbot de Comercio Internacional en funcionamiento!",
        "status": "running",
        "services": {
            "pinecone": "✅" if pinecone_index else "❌",
            "groq": "✅" if groq_client else "❌"
        }
    })

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.json
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Mensaje de usuario no proporcionado."}), 400

        rag_context = retrieve_from_pinecone(user_message)
        
        prompt_template = f"""
        Eres un experto en comercio internacional. 
        Utiliza el siguiente contexto para responder a las preguntas de los usuarios. 
        Si no encuentras información relevante en el contexto, simplemente responde que no puedes ayudar con esa pregunta específica.

        Contexto: {rag_context}

        Pregunta: {user_message}
        """

        response = generate_response(prompt_template)

        return jsonify({"response": response})
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

# Para Vercel, necesitamos exportar la app
handler = app