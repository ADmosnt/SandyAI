# app/services/audio/speaker_service.py

import os
import pickle
import uuid
from datetime import datetime
from pathlib import Path

import re
import unicodedata

import numpy as np
import torch
import soundfile as sf
import torchaudio
from dotenv import load_dotenv

# --- Patch: torchaudio nightly removed list_audio_backends(), speechbrain still needs it ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# --- Patch: bypass torchaudio's torchcodec dispatch entirely ---
# torchaudio nightly (2.11+) forces torchcodec which needs FFmpeg DLLs.
# We replace torchaudio.load with a direct soundfile implementation.
def _direct_sf_load(filepath, frame_offset=0, num_frames=-1,
                    normalize=True, channels_first=True, **_kw):
    start = frame_offset if frame_offset > 0 else 0
    stop = (start + num_frames) if num_frames > 0 else None
    data, samplerate = sf.read(str(filepath), start=start, stop=stop,
                               dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.copy())   # (samples, channels)
    if channels_first:
        tensor = tensor.T                    # (channels, samples)
    return tensor, samplerate

torchaudio.load = _direct_sf_load

from speechbrain.inference import SpeakerRecognition

load_dotenv()

VOICES_DB_PATH = "sandy_voices_db"
SPEAKERS_FILE = Path(VOICES_DB_PATH) / "speakers.pkl"
LOCAL_MODEL_PATH = Path(VOICES_DB_PATH) / "model_source"

# Umbral mínimo para "reconocer" a alguien (tu valor original)
RECOGNIZE_THRESHOLD = 0.25

# Umbral para "auto-aprender" una nueva muestra cuando ya te reconoció
# Debe ser > RECOGNIZE_THRESHOLD
LEARN_THRESHOLD = 0.35

# Límite de muestras por persona
MAX_EMBEDDINGS_PER_SPEAKER = 20

# Evita guardar duplicados: si el nuevo embedding es demasiado similar a uno existente, no lo guardes.
# OJO: este valor asume cos_sim en [0..1]. Si tu cos_sim se mueve distinto, baja/sube.
DUPLICATE_SIM_THRESHOLD = 0.98


class SpeakerService:
    def __init__(self):
        print("🎙️ [Voces] Inicializando reconocimiento de hablantes (MODO CPU SEGURO)...")

        self.recognizer = None
        self.device = torch.device("cpu")

        try:
            if LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / "hyperparams.yaml").exists():
                print("   ↳ Cargando desde disco local...")
                self.recognizer = SpeakerRecognition.from_hparams(
                    source=str(LOCAL_MODEL_PATH),
                    savedir="tmp_speaker_models",
                    run_opts={"device": "cpu"},
                )
            else:
                print("⚠️ Modelo local no encontrado. Intentando fallback online...")
                self.recognizer = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="tmp_speaker_models",
                    run_opts={"device": "cpu"},
                )

            print("   ↳ Modelo listo en CPU (Estabilidad Máxima).")

        except OSError as e:
            print(f"❌ Error de Sistema (WinError): {e}")
        except Exception as e:
            print(f"❌ Error crítico SpeechBrain: {e}")

        Path(VOICES_DB_PATH).mkdir(exist_ok=True)

        self.registered_speakers = self._load_speakers()
        self.unknown_voice_encounters = {}

        # Migración automática (embedding -> embeddings)
        migrated = self._migrate_storage_format()
        if migrated:
            self._save_speakers()

        if self.recognizer:
            print(f"✅ [Voces] {len(self.registered_speakers)} voces registradas")

    # -----------------------------
    # Persistencia
    # -----------------------------
    def _load_speakers(self):
        if SPEAKERS_FILE.exists():
            try:
                with open(SPEAKERS_FILE, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def _save_speakers(self):
        with open(SPEAKERS_FILE, "wb") as f:
            pickle.dump(self.registered_speakers, f)

    def _migrate_storage_format(self) -> bool:
        """
        Migra:
          {'embedding': np.array(...)} -> {'embeddings': [np.array(...)]}
        Retorna True si hizo cambios.
        """
        changed = False
        for spk_id, data in list(self.registered_speakers.items()):
            if not isinstance(data, dict):
                continue

            if "embeddings" in data and isinstance(data["embeddings"], list):
                # ok
                continue

            if "embedding" in data and data["embedding"] is not None:
                data["embeddings"] = [data["embedding"]]
                data.pop("embedding", None)
                self.registered_speakers[spk_id] = data
                changed = True
            else:
                # estructura rara: normaliza igual
                data["embeddings"] = []
                data.pop("embedding", None)
                self.registered_speakers[spk_id] = data
                changed = True

        return changed

    # -----------------------------
    # Embeddings
    # -----------------------------
    def extract_voice_embedding(self, audio_path: str):
        if self.recognizer is None:
            return None
        try:
            signal = self.recognizer.load_audio(audio_path)
            embedding = self.recognizer.encode_batch(signal.unsqueeze(0))
            return embedding.squeeze().numpy()
        except Exception as e:
            print(f"⚠️ Error embedding: {e}")
            return None

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    def _best_score_against_embeddings(self, emb: np.ndarray, emb_list: list) -> float:
        best = -1.0
        for e in emb_list:
            try:
                s = self._cos_sim(emb, e)
            except Exception:
                continue
            if s > best:
                best = s
        return best

    def _maybe_learn_embedding(self, spk_id: str, new_emb: np.ndarray, score: float) -> None:
        """
        Auto-aprendizaje:
        - solo si hay cupo
        - solo si no es duplicado
        """
        data = self.registered_speakers.get(spk_id)
        if not isinstance(data, dict):
            return

        emb_list = data.get("embeddings", [])
        if not isinstance(emb_list, list):
            emb_list = []
            data["embeddings"] = emb_list

        if len(emb_list) >= MAX_EMBEDDINGS_PER_SPEAKER:
            return

        # si es casi idéntico a algo existente, no lo guardes
        best_existing = self._best_score_against_embeddings(new_emb, emb_list) if emb_list else -1.0
        if best_existing >= DUPLICATE_SIM_THRESHOLD:
            return

        emb_list.append(new_emb)
        data["embeddings"] = emb_list
        data["last_learned"] = datetime.now().isoformat()
        data["learned_count"] = int(data.get("learned_count", 0)) + 1
        self.registered_speakers[spk_id] = data
        self._save_speakers()

    # -----------------------------
    # API pública
    # -----------------------------
    def identify_speaker(self, audio_path: str):
        """
        Retorna:
          (speaker_name, speaker_id, confidence, is_new)

        - Si NO reconoce: ("stranger", None, best_score, True)
        - Si reconoce: (name, spk_id, best_score, False)

        Auto-aprendizaje:
          Si best_score >= LEARN_THRESHOLD, guarda el embedding como otra muestra de ese mismo usuario.
        """
        if self.recognizer is None:
            return ("unknown", None, 0.0, False)

        embedding = self.extract_voice_embedding(audio_path)
        if embedding is None:
            return ("unknown", None, 0.0, False)

        best_match = None
        best_score = -1.0

        # Asegura que esté migrado (por si speakers.pkl viejo se cargó)
        self._migrate_storage_format()

        for spk_id, data in self.registered_speakers.items():
            if not isinstance(data, dict):
                continue
            emb_list = data.get("embeddings", [])
            if not emb_list:
                continue

            score = self._best_score_against_embeddings(embedding, emb_list)
            if score > best_score:
                best_score = score
                best_match = (data.get("name", "unknown"), spk_id)

        if best_match and best_score > RECOGNIZE_THRESHOLD:
            name, spk_id = best_match

            # auto-learn si el score fue alto
            if best_score >= LEARN_THRESHOLD:
                self._maybe_learn_embedding(spk_id, embedding, best_score)

            return (name, spk_id, best_score, False)

        return ("stranger", None, best_score, True)

    def register_speaker(self, audio_path: str, name: str):
        """
        Registra una nueva persona (nuevo spk_id) con su primer embedding.
        """
        if self.recognizer is None:
            return False

        embedding = self.extract_voice_embedding(audio_path)
        if embedding is None:
            return False

        spk_id = str(uuid.uuid4())
        self.registered_speakers[spk_id] = {
            "name": name,
            "embeddings": [embedding],
            "date": datetime.now().isoformat(),
            "learned_count": 0,
        }
        self._save_speakers()
        return spk_id
    
    def add_embedding_to_speaker(self, spk_id: str, audio_path: str) -> bool:
        """
        Fuerza agregar un embedding adicional al speaker existente.
        Usa la misma lógica anti-duplicados y límite de MAX_EMBEDDINGS_PER_SPEAKER.
        """
        try:
            if self.recognizer is None:
                return False

            self._migrate_storage_format()

            data = self.registered_speakers.get(spk_id)
            if not isinstance(data, dict):
                return False

            new_emb = self.extract_voice_embedding(audio_path)
            if new_emb is None:
                return False

            emb_list = data.get("embeddings", [])
            if not isinstance(emb_list, list):
                emb_list = []
                data["embeddings"] = emb_list

            if len(emb_list) >= MAX_EMBEDDINGS_PER_SPEAKER:
                return False

            best_existing = self._best_score_against_embeddings(new_emb, emb_list) if emb_list else -1.0
            if best_existing >= DUPLICATE_SIM_THRESHOLD:
                return False

            emb_list.append(new_emb)
            data["embeddings"] = emb_list
            data["last_manual_learned"] = datetime.now().isoformat()
            data["learned_count"] = int(data.get("learned_count", 0)) + 1
            self.registered_speakers[spk_id] = data
            self._save_speakers()
            return True
        except Exception:
            return False


    def get_curious_strangers(self, min_encounters: int = 3):
        """
        (Opcional) Mantengo el método por compatibilidad,
        pero tu lógica actual no está llenando unknown_voice_encounters.
        """
        curious = []
        for voice_id, data in self.unknown_voice_encounters.items():
            if data.get("count", 0) >= min_encounters:
                curious.append(
                    {
                        "voice_id": voice_id,
                        "encounters": data["count"],
                        "first_heard": data.get("first_heard"),
                    }
                )
        return curious
    

    def get_top_candidates(self, audio_path: str, k: int = 3):
        """
        Retorna top-K candidatos: list[{"name": str, "speaker_id": str, "score": float}]
        Ordenados desc por score. Útil cuando identify_speaker da 'stranger'.
        """
        if self.recognizer is None:
            return []

        emb = self.extract_voice_embedding(audio_path)
        if emb is None:
            return []

        self._migrate_storage_format()

        scored = []
        for spk_id, data in self.registered_speakers.items():
            if not isinstance(data, dict):
                continue
            emb_list = data.get("embeddings", [])
            if not emb_list:
                continue
            score = self._best_score_against_embeddings(emb, emb_list)
            scored.append({
                "name": data.get("name", "unknown"),
                "speaker_id": spk_id,
                "score": float(score),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(1, int(k))]


    def normalize_person_token(self, text: str) -> str:
        """
        Normaliza texto para comparar nombres:
        - lower
        - quita tildes/diacríticos con unicode
        - preserva letras/números/espacios y también ' y - (útiles en nombres)
        """
        if not text:
            return ""

        t = text.strip().lower()
        t = unicodedata.normalize("NFKD", t)
        t = "".join(c for c in t if not unicodedata.combining(c))
        t = re.sub(r"[^a-z0-9\s'\-]", " ", t)  # deja ' y - para O'Connor, Jean-Luc
        t = re.sub(r"\s+", " ", t).strip()
        return t
