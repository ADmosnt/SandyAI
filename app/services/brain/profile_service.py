# app/services/brain/profile_service.py 

import json
import os
from datetime import datetime
from google import genai
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

load_dotenv()

PROFILES_DIR = "sandy_user_profiles"

DEFAULT_PROFILE = {
    "speaker_id": None,
    "user": {
        "primary_name": None,  # Nombre principal
        "alternate_names": [],  # ["Alan Gabriel", "Alan G."]
        "nicknames": [],  # ["Gabo", "Alangel"]
        "voice_registered": False,
        "first_interaction": None,
        "total_interactions": 0,
        "location": None,
        "occupation": None,
        "interests": [],
        "projects": []
    },
    "social": {
        "known_associates": {},  # {"velok": "santiago_speaker_id"}
        "referred_by_others_as": []  # Cómo otros lo llaman
    },
    "preferences": {
        "communication_style": "conversational",
        "detail_level": "medium",
        "language": "es",
        "topics_to_avoid": []
    },
    "facts": {},
    "relationship": {
        "trust_level": 0,
        "interaction_frequency": "new"
    },
    "last_updated": None
}

class MultiUserProfileService:
    def __init__(self):
        Path(PROFILES_DIR).mkdir(exist_ok=True)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.current_speaker_id = None
        self.profiles = {}
        print(f"👥 [Perfiles] Sistema multi-usuario activo")

    def set_current_speaker(self, speaker_id: str, speaker_name: str = None):
        """Establece el hablante actual"""
        self.current_speaker_id = speaker_id
        
        if speaker_id not in self.profiles:
            self.profiles[speaker_id] = self._load_profile(speaker_id)
            
            if self.profiles[speaker_id]['user']['first_interaction'] is None:
                self.profiles[speaker_id]['speaker_id'] = speaker_id
                self.profiles[speaker_id]['user']['first_interaction'] = datetime.now().isoformat()
                if speaker_name:
                    self.profiles[speaker_id]['user']['primary_name'] = speaker_name
                    self.profiles[speaker_id]['user']['voice_registered'] = True

    def _get_profile_path(self, speaker_id: str):
        return Path(PROFILES_DIR) / f"profile_{speaker_id}.json"

    def _load_profile(self, speaker_id: str):
        profile_path = self._get_profile_path(speaker_id)
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
                # Migrar perfiles antiguos
                if 'social' not in profile:
                    profile['social'] = DEFAULT_PROFILE['social'].copy()
                if 'primary_name' not in profile['user']:
                    # Migrar de 'name' a 'primary_name'
                    profile['user']['primary_name'] = profile['user'].get('name')
                    profile['user']['alternate_names'] = []
                    profile['user']['nicknames'] = []
                return profile
        return DEFAULT_PROFILE.copy()

    def _save_profile(self, speaker_id: str = None):
        if speaker_id is None:
            speaker_id = self.current_speaker_id
        
        if speaker_id not in self.profiles:
            return
        
        self.profiles[speaker_id]['last_updated'] = datetime.now().isoformat()
        self.profiles[speaker_id]['user']['total_interactions'] += 1
        
        profile_path = self._get_profile_path(speaker_id)
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(self.profiles[speaker_id], f, indent=2, ensure_ascii=False)

    def get_display_name(self, speaker_id: str) -> str:
        """Obtiene el nombre de display del usuario"""
        if speaker_id not in self.profiles:
            return "Usuario"
        
        user = self.profiles[speaker_id]['user']
        return user.get('primary_name') or "Usuario"

    def add_alias(self, speaker_id: str, alias: str):
        """
        Agregar nombre alternativo o apodo.
        Decide automáticamente si es nombre o apodo por longitud.
        """
        if speaker_id not in self.profiles:
            return
        
        profile = self.profiles[speaker_id]
        alias = alias.strip()
        
        # Si tiene espacios o múltiples palabras → nombre alternativo
        # Si es una palabra → apodo
        if len(alias.split()) > 1:
            if alias not in profile['user']['alternate_names']:
                profile['user']['alternate_names'].append(alias)
                print(f"👤 [Perfil] Nombre alternativo agregado: {alias}")
        else:
            if alias not in profile['user']['nicknames']:
                profile['user']['nicknames'].append(alias)
                print(f"🏷️  [Perfil] Apodo agregado: {alias}")
        
        self._save_profile(speaker_id)

    def link_nickname_to_speaker(self, nickname: str, target_speaker_id: str, used_by_speaker_id: str):
        """
        Asociar un apodo con un speaker específico.
        Esto permite que "Velok" se resuelva a "Santiago" cuando Gabriel lo menciona.
        """
        if used_by_speaker_id not in self.profiles:
            return
        
        nickname_lower = nickname.lower()
        self.profiles[used_by_speaker_id]['social']['known_associates'][nickname_lower] = target_speaker_id
        self._save_profile(used_by_speaker_id)
        
        target_name = self.get_display_name(target_speaker_id)
        print(f"🔗 [Social] '{nickname}' → {target_name} (según {self.get_display_name(used_by_speaker_id)})")

    def resolve_name(self, name: str, context_speaker_id: str = None) -> Optional[str]:
        """
        Resolver un nombre/apodo a un speaker_id.
        
        Args:
            name: Nombre o apodo a buscar
            context_speaker_id: Quién está hablando (para resolver apodos personales)
        
        Returns:
            speaker_id si se encuentra, None si no
        """
        name_lower = name.lower().strip()
        
        # Buscar en todos los perfiles
        for speaker_id, profile in self.profiles.items():
            user = profile['user']
            
            # Nombre principal
            if user['primary_name'] and user['primary_name'].lower() == name_lower:
                return speaker_id
            
            # Nombres alternativos
            if any(alt.lower() == name_lower for alt in user.get('alternate_names', [])):
                return speaker_id
            
            # Apodos
            if any(nick.lower() == name_lower for nick in user.get('nicknames', [])):
                return speaker_id
        
        # Buscar en asociados del hablante actual
        if context_speaker_id and context_speaker_id in self.profiles:
            associates = self.profiles[context_speaker_id]['social'].get('known_associates', {})
            if name_lower in associates:
                return associates[name_lower]
        
        return None

    def get_current_context(self) -> str:
        """Contexto del hablante actual para Gemini"""
        if not self.current_speaker_id or self.current_speaker_id not in self.profiles:
            return "USUARIO: Desconocido (primera interacción)\n"
        
        profile = self.profiles[self.current_speaker_id]
        user = profile['user']
        
        # Construir nombre completo con aliases
        names = [user.get('primary_name', 'Usuario')]
        if user.get('alternate_names'):
            names.extend(user['alternate_names'])
        if user.get('nicknames'):
            names.extend(user['nicknames'])
        
        context = f"HABLANTE ACTUAL: {names[0]}\n"
        if len(names) > 1:
            context += f"- También conocido como: {', '.join(names[1:])}\n"
        
        context += f"- Interacciones previas: {user['total_interactions']}\n"
        
        if user['interests']:
            context += f"- Intereses: {', '.join(user['interests'])}\n"
        
        if profile['facts']:
            context += "\nHECHOS REGISTRADOS:\n"
            for key, value in profile['facts'].items():
                context += f"- {key}: {value}\n"
        
        # Asociados conocidos
        if profile['social'].get('known_associates'):
            context += "\nPERSONAS CONOCIDAS:\n"
            for nickname, sid in profile['social']['known_associates'].items():
                known_name = self.get_display_name(sid)
                context += f"- {nickname.capitalize()} = {known_name}\n"
        
        return context

    def extract_facts(self, conversation_history: list[str]):
        """Extrae hechos del historial (igual que antes)"""
        if not self.current_speaker_id or len(conversation_history) < 4:
            return
        
        extraction_prompt = f"""
            Analiza esta conversación y extrae información NUEVA sobre el usuario.
            Responde SOLO en JSON válido:
            {{
            "primary_name": "string o null",
            "alternate_names": ["lista de nombres adicionales"],
            "nicknames": ["lista de apodos"],
            "interests": ["lista"],
            "projects": ["lista"],
            "facts": {{"clave": "valor"}},
            "preferences": {{
                "communication_style": "formal/conversational/technical",
                "detail_level": "low/medium/high"
            }}
            }}

            Si no hay info nueva, devuelve campos vacíos/null.

            CONVERSACIÓN:
            {chr(10).join(conversation_history[-10:])}

            JSON:
            """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=extraction_prompt,
                config={"temperature": 0.1}
            )
            
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1].replace("json", "", 1).strip()
            
            extracted = json.loads(raw_text)
            self._merge_extracted_data(extracted)
            self._save_profile()
            print("🧠 [Perfil] Actualizado con nueva información")
            
        except Exception as e:
            print(f"⚠️ [Perfil] Error extrayendo hechos: {e}")

    def _merge_extracted_data(self, extracted: dict):
        """Mezcla datos extraídos con perfil actual"""
        profile = self.profiles[self.current_speaker_id]
        
        if extracted.get('primary_name'):
            profile['user']['primary_name'] = extracted['primary_name']
        
        # Merge de nombres alternativos
        if extracted.get('alternate_names'):
            current = set(profile['user'].get('alternate_names', []))
            new = set(extracted['alternate_names'])
            profile['user']['alternate_names'] = list(current | new)
        
        # Merge de apodos
        if extracted.get('nicknames'):
            current = set(profile['user'].get('nicknames', []))
            new = set(extracted['nicknames'])
            profile['user']['nicknames'] = list(current | new)
        
        # Intereses
        if extracted.get('interests'):
            current = set(profile['user']['interests'])
            new = set(extracted['interests'])
            profile['user']['interests'] = list(current | new)
        
        # Proyectos
        if extracted.get('projects'):
            current = set(profile['user']['projects'])
            new = set(extracted['projects'])
            profile['user']['projects'] = list(current | new)
        
        # Facts
        if extracted.get('facts'):
            profile['facts'].update(extracted['facts'])
        
        # Preferencias
        if extracted.get('preferences'):
            profile['preferences'].update(extracted['preferences'])

    def set_primary_name(self, speaker_id: str, new_name: str):
        """
        Setea primary_name sin perder el anterior:
        - si había un primary diferente, lo guarda como alias (nick o alternate)
        """
        if speaker_id not in self.profiles:
            self.profiles[speaker_id] = self._load_profile(speaker_id)

        profile = self.profiles[speaker_id]
        old = profile['user'].get('primary_name')

        new_name = (new_name or "").strip()
        if not new_name:
            return

        if old and old.strip() and old.strip() != new_name:
            # preserva el anterior como alias
            self.add_alias(speaker_id, old.strip())

        profile['user']['primary_name'] = new_name
        self._save_profile(speaker_id)
