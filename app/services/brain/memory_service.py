# app/services/brain/memory_service.py

import chromadb
import os
import uuid
from google import genai
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class MemoryService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        
        # ChromaDB persistente
        self.chroma_client = chromadb.PersistentClient(path="sandy_memory_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="episodic_memory",
            metadata={"description": "Conversaciones pasadas de Sandy"}
        )
        
        count = self.collection.count()
        print(f"🧠 [Memoria] Recuerdos cargados: {count}")

    def _get_embedding(self, text: str):
        """Genera embedding usando Gemini"""
        try:
            result = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"⚠️ Error generando embedding: {e}")
            return None

    def add_memory(self, text: str, metadata: dict = None):
        """Guarda un recuerdo con metadatos"""
        # Filtrar contenido trivial
        if len(text.strip()) < 20:
            return  # Ignorar mensajes muy cortos
        
        if any(trivial in text.lower() for trivial in ["ok", "vale", "gracias", "de nada"]):
            return  # Ignorar conversaciones triviales
        
        vector = self._get_embedding(text)
        if not vector:
            return

        if metadata is None:
            metadata = {}
        
        metadata['timestamp'] = datetime.now().isoformat()
        
        self.collection.add(
            documents=[text],
            embeddings=[vector],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def search_memory(self, query: str, n_results: int = 3):
        """Busca recuerdos relevantes"""
        vector = self._get_embedding(query)
        if not vector:
            return []

        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        memories = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                # Solo incluir si la similitud es alta (distancia baja)
                if distance < 0.7:  # Threshold ajustable
                    memories.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'relevance': 1 - distance
                    })
        
        return memories

    def summarize_and_consolidate(self, conversation_chunk: list[str]):
        """
        Usa Gemini para resumir un trozo de conversación antes de guardarlo.
        Esto evita guardar 10,000 mensajes basura.
        """
        if len(conversation_chunk) < 3:
            return
        
        # Unir la conversación
        text = "\n".join(conversation_chunk)
        
        # Prompt para resumir
        summary_prompt = f"""
            Resume esta conversación en 2-3 oraciones, extrayendo solo la información IMPORTANTE.
            Ignora saludos, despedidas y conversación trivial.
            Si no hay nada importante, responde "SKIP".

            CONVERSACIÓN:
            {text}

            RESUMEN:
            """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=summary_prompt,
                config={"temperature": 0.3}
            )
            
            summary = response.text.strip()
            
            if summary != "SKIP" and len(summary) > 20:
                self.add_memory(
                    summary,
                    metadata={"type": "conversation_summary", "original_length": len(conversation_chunk)}
                )
                print(f"💾 [Memoria] Resumen consolidado: '{summary[:50]}...'")
                
        except Exception as e:
            print(f"⚠️ [Memoria] Error consolidando: {e}")