# app/services/audio/listening_service.py

from typing import Optional

import pyaudio
import wave
import uuid
import numpy as np
from faster_whisper import WhisperModel
import torch
import os
import time
from dotenv import load_dotenv

try:
    from app.services.audio.speaker_service import SpeakerService
    SPEAKER_ID_AVAILABLE = True
except ImportError:
    SPEAKER_ID_AVAILABLE = False
    print("⚠️ Speaker identification no disponible (falta SpeakerService)")

load_dotenv()


from app.services.audio.tts_service import _check_cuda_or_die, CUDAIncompatibleError


CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_DURATION = 1.0

MIC_INDEX = int(os.getenv("MIC_INDEX")) if os.getenv("MIC_INDEX") else None
THRESHOLD = int(os.getenv("MIC_THRESHOLD", "500"))
ENABLE_SPEAKER_ID = os.getenv("ENABLE_SPEAKER_ID", "0") == "1"


class ListeningService:
    def __init__(self):
        print(f"🎧 Configurando Oído... (Mic ID: {MIC_INDEX} | Threshold: {THRESHOLD})")

        _check_cuda_or_die()
        try:
            self.model = WhisperModel("medium", device="cuda", compute_type="float16")
            print("✅ Modelo Whisper cargado en CUDA.")
        except Exception as e:
            print(f"❌ Error Whisper: {e}")
            raise

        self.audio = pyaudio.PyAudio()
        self.stream = None

        self.speaker_service = None
        self.primary_user_id = None

        self.last_voice_sample_path = None  # type: Optional[str]

        if ENABLE_SPEAKER_ID and SPEAKER_ID_AVAILABLE:
            try:
                self.speaker_service = SpeakerService()
                print("✅ [Speaker ID] Sistema de reconocimiento de voz activo")
            except Exception as e:
                print(f"⚠️ [Speaker ID] No se pudo inicializar: {e}")

    def listen_and_transcribe(self, allow_unknown: bool = False, return_meta: bool = False):
        """
        Returns (por defecto):
          - tuple(texto, speaker_id) si speaker ID está activo
          - str solo texto si speaker ID está desactivado

        Si return_meta=True y speaker ID está activo:
          - (texto, speaker_id, meta)
            meta = {
              "speaker_name": str|None,
              "confidence": float|None,
              "audio_path": str|None,
              "candidates": list,
              "avg_volume": float|None,
              "max_volume": float|None,
              "duration_s": float|None
            }

        allow_unknown=True se usa cuando estás esperando introducción,
        para NO filtrar strangers.
        """
        self.last_voice_sample_path = None

        # 1) GRABACIÓN con VAD
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=MIC_INDEX,
                frames_per_buffer=CHUNK
            )
        except Exception as e:
            print(f"❌ Error abriendo micrófono ID {MIC_INDEX}: {e}")
            return self._return_format(None, None, return_meta=return_meta, meta=None)

        print("🎤 ...", end="", flush=True)
        frames = []
        silence_counter = 0
        started_talking = False

        chunks_per_second = RATE / CHUNK
        max_silence_chunks = int(chunks_per_second * SILENCE_DURATION)

        # métricas para gating barge-in
        vol_sum = 0.0
        vol_n = 0
        vol_max = 0.0
        t_start = time.perf_counter()

        speech_start_t = None

        while True:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = float(np.abs(audio_data).mean())

                vol_sum += volume
                vol_n += 1
                if volume > vol_max:
                    vol_max = volume

                if volume > THRESHOLD:
                    silence_counter = 0
                    if not started_talking:
                        print("🔊", end="", flush=True)
                        started_talking = True
                        speech_start_t = time.perf_counter()
                else:
                    if started_talking:
                        silence_counter += 1

                if started_talking and silence_counter > max_silence_chunks:
                    break
            except KeyboardInterrupt:
                raise

        t_end = time.perf_counter()

        self.stream.stop_stream()
        self.stream.close()
        print(" ")

        if not frames:
            return self._return_format("", None, return_meta=return_meta, meta=None)

        audio_buffer = b"".join(frames)
        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

        avg_volume = (vol_sum / vol_n) if vol_n else None
        base = speech_start_t if speech_start_t is not None else t_start
        duration_s = max(0.0, float(t_end - base))


        # 3) SPEAKER ID (opcional)
        speaker_id = None
        meta = {
            "speaker_name": None,
            "confidence": None,
            "audio_path": None,
            "candidates": [],
            "avg_volume": float(avg_volume) if avg_volume is not None else None,
            "max_volume": float(vol_max) if vol_max is not None else None,
            "duration_s": float(duration_s) if duration_s is not None else None,
            "captured_at": time.time(),
        }

        if self.speaker_service is not None:
            temp_audio_path = f"temp_voice_sample_{os.getpid()}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.wav"
            self._save_audio_wav(audio_buffer, temp_audio_path)

            speaker_name, identified_id, confidence, is_new = self.speaker_service.identify_speaker(temp_audio_path)

            # Guardar SIEMPRE el wav para main (confirmación / aprendizaje / enrollment / barge-in)
            self.last_voice_sample_path = temp_audio_path

            meta["speaker_name"] = speaker_name
            meta["confidence"] = float(confidence) if confidence is not None else None
            meta["audio_path"] = temp_audio_path

            # Top candidatos (útil para confirmación)
            try:
                meta["candidates"] = self.speaker_service.get_top_candidates(temp_audio_path, k=3)
            except Exception:
                meta["candidates"] = []

            if speaker_name == "stranger":
                print(f"🔊 [stranger | conf={confidence:.2f}]", end=" ")
                speaker_id = None
            else:
                print(f"🔊 [{speaker_name} | conf={confidence:.2f}]", end=" ")
                speaker_id = identified_id

        # 4) TRANSCRIPCIÓN
        segments, _ = self.model.transcribe(
            audio_np,
            beam_size=5,
            language="es",
            vad_filter=True
        )
        full_text = " ".join([segment.text for segment in segments]).strip()

        return self._return_format(full_text, speaker_id, return_meta=return_meta, meta=meta)

    def _return_format(self, text, speaker_id, return_meta: bool = False, meta=None):
        if self.speaker_service is not None:
            if return_meta:
                return text, speaker_id, meta
            return text, speaker_id
        return text

    def _save_audio_wav(self, audio_data, filename):
        wf = wave.open(filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
        wf.close()

    def register_primary_user(self, name: str):
        if self.speaker_service is None:
            print("⚠️ Speaker ID no está activo. No se puede registrar.")
            return False

        print(f"🎙️ Registrando voz de {name}...")
        print("Por favor, di una frase (mínimo 3 segundos)...")

        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=MIC_INDEX,
                frames_per_buffer=CHUNK
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

        frames = []
        silence_counter = 0
        started_talking = False
        chunks_per_second = RATE / CHUNK
        max_silence_chunks = int(chunks_per_second * SILENCE_DURATION)

        print("🎤 Grabando...", end="", flush=True)

        while True:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()

            if volume > THRESHOLD:
                silence_counter = 0
                if not started_talking:
                    print("🔊", end="", flush=True)
                    started_talking = True
            else:
                if started_talking:
                    silence_counter += 1

            if started_talking and silence_counter > max_silence_chunks:
                break

        self.stream.stop_stream()
        self.stream.close()
        print(" ✅")

        audio_buffer = b"".join(frames)
        temp_path = "temp_enrollment.wav"
        self._save_audio_wav(audio_buffer, temp_path)

        spk_id = self.speaker_service.register_speaker(temp_path, name)

        if spk_id:
            self.primary_user_id = spk_id
            print(f"✅ Voz de {name} registrada correctamente")

        try:
            os.remove(temp_path)
        except Exception:
            pass

        return bool(spk_id)

    def __del__(self):
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        try:
            self.audio.terminate()
        except Exception:
            pass
