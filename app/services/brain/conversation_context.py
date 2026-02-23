# app/services/brain/conversation_context.py

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class ConversationContext:
    """
    Maneja el estado de la conversación y expectativas.
    Permite a Sandy "recordar" qué espera en el siguiente turno.
    """
    # Expectativa de presentación
    expecting_introduction: bool = False
    expected_name: Optional[str] = None
    expectation_timeout: Optional[datetime] = None
    
    # Tracking de conversación
    last_speaker_id: Optional[str] = None
    conversation_participants: list = field(default_factory=list)
    
    def expect_introduction(self, name: str, timeout_seconds: int = 60):
        """
        Sandy está esperando que la próxima voz desconocida sea 'name'.
        
        Args:
            name: Nombre esperado
            timeout_seconds: Cuánto tiempo esperar antes de cancelar
        """
        self.expecting_introduction = True
        self.expected_name = name
        self.expectation_timeout = datetime.now()
        print(f"🔔 [Contexto] Esperando conocer a '{name}' (timeout: {timeout_seconds}s)")
    
    def clear_expectation(self):
        """Limpiar expectativa de presentación"""
        if self.expecting_introduction:
            print(f"✅ [Contexto] Expectativa cumplida: {self.expected_name}")
        self.expecting_introduction = False
        self.expected_name = None
        self.expectation_timeout = None
    
    def check_timeout(self, timeout_seconds: int = 60) -> bool:
        """
        Verifica si la expectativa expiró.
        Retorna True si expiró.
        """
        if not self.expecting_introduction or not self.expectation_timeout:
            return False
        
        elapsed = (datetime.now() - self.expectation_timeout).total_seconds()
        if elapsed > timeout_seconds:
            print(f"⏰ [Contexto] Timeout: Expectativa de '{self.expected_name}' expiró")
            self.clear_expectation()
            return True
        
        return False
    
    def add_participant(self, speaker_id: str):
        """Agregar participante activo a la conversación"""
        if speaker_id and speaker_id not in self.conversation_participants:
            self.conversation_participants.append(speaker_id)
    
    def set_last_speaker(self, speaker_id: str):
        """Actualizar último hablante"""
        self.last_speaker_id = speaker_id
    
    def reset(self):
        """Resetear todo el contexto (nueva conversación)"""
        self.expecting_introduction = False
        self.expected_name = None
        self.expectation_timeout = None
        self.last_speaker_id = None
        self.conversation_participants = []
        print("🔄 [Contexto] Reiniciado")

# Instancia global
conversation_ctx = ConversationContext()