# app/services/gemini.py

import os
import re
import threading
from collections import deque
from dotenv import load_dotenv
from google import genai
from google.genai import types
from app.core.personality import build_system_prompt

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "0") == "1"
ENABLE_MULTI_PROFILES = os.getenv("ENABLE_MULTI_PROFILES", "0") == "1"

# Importar servicios opcionales
memory = None
profiles = None

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"⚠️ Error inicializando cliente Gemini: {e}")
    client = None

if ENABLE_MEMORY:
    try:
        from app.services.brain.memory_service import MemoryService
        memory = MemoryService()
    except BaseException as e:
        print(f"⚠️ Memoria episódica desactivada (ChromaDB falló): {e}")
        memory = None

if ENABLE_MULTI_PROFILES:
    try:
        from app.services.brain.profile_service import MultiUserProfileService
        profiles = MultiUserProfileService()
    except BaseException as e:
        print(f"⚠️ Perfiles multi-usuario desactivados: {e}")
        profiles = None

# Memoria a corto plazo
history_chat: deque[str] = deque(maxlen=10)

# Contador para extracciones periódicas
_turn_counter = 0
EXTRACT_EVERY_N_TURNS = 5

# System prompt generado desde personality.json
SYSTEM_PROMPT = build_system_prompt()

def query_gemini_stream(text: str, speaker_id: str = None):
    """
    Genera respuesta con streaming.
    
    Args:
        text: Query del usuario
        speaker_id: ID del hablante (opcional, para multi-user)
    """
    global _turn_counter
    
    try:
        # 1. PERFIL (si está activo)
        profile_context = ""
        if profiles and speaker_id:
            profiles.set_current_speaker(speaker_id)
            profile_context = profiles.get_current_context()
        
        # 2. MEMORIA EPISÓDICA (si está activa)
        memory_context = ""
        if memory:
            memories = memory.search_memory(text, n_results=2)
            if memories:
                memory_context = "\nRECUERDOS RELEVANTES:\n"
                for mem in memories:
                    memory_context += f"- {mem['content']}\n"
        
        # 3. HISTORIAL RECIENTE
        recent_context = "\n".join(history_chat)
        
        # 4. PROMPT FINAL
        full_prompt = f"""{SYSTEM_PROMPT}

{profile_context}

{memory_context}

HISTORIAL RECIENTE:
{recent_context}

Usuario dice: {text}
Sandy:"""
        
        # Consolidación antes de agregar (si memoria está activa)
        if memory and len(history_chat) == history_chat.maxlen:
            chunk_to_consolidate = list(history_chat)[:3]
            threading.Thread(
                target=memory.summarize_and_consolidate,
                args=(chunk_to_consolidate,),
                daemon=True
            ).start()
        
        history_chat.append(f"Tú: {text}")
        
        # Generación con streaming
        response = client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=250
            )
        )
        
        buffer = ""
        full_response_text = ""

        for chunk in response:
            if chunk.text:
                part = chunk.text
                buffer += part
                full_response_text += part

                sentences = re.split(r'(?<=[.?!:])\s+', buffer)
                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        if sentence.strip():
                            yield sentence.strip()
                    buffer = sentences[-1]

        if buffer.strip():
            yield buffer.strip()

        history_chat.append(f"Sandy: {full_response_text}")
        
        # Guardar en memoria (si está activa)
        if memory:
            threading.Thread(
                target=memory.add_memory,
                args=(f"Usuario: {text} | Sandy: {full_response_text}",),
                daemon=True
            ).start()
        
        # Extracción periódica de hechos (si profiles está activo)
        if profiles and speaker_id:
            _turn_counter += 1
            if _turn_counter >= EXTRACT_EVERY_N_TURNS:
                _turn_counter = 0
                threading.Thread(
                    target=profiles.extract_facts,
                    args=(list(history_chat),),
                    daemon=True
                ).start()

    except Exception as e:
        print(f"❌ Error Gemini: {e}")
        yield "Error en mi sistema de memoria."