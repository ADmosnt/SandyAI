# app/services/audio/tts_service.py

import time
import os
import threading
import queue
import re
import numpy as np
import sounddevice as sd
import queue as pyqueue
import torch
from dotenv import load_dotenv
from TTS.api import TTS

load_dotenv()

# =========================
# Torch 2.6+ allowlist (PyTorch 2.7.1) para cargar XTTS
# =========================
def _allowlist_xtts_globals():
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig

        torch.serialization.add_safe_globals([
            XttsConfig,
            XttsAudioConfig,
            XttsArgs,
            BaseDatasetConfig,
        ])
    except Exception as e:
        print(f"⚠️ Allowlist XTTS falló: {e}")

_allowlist_xtts_globals()

# =========================
# Config
# =========================

TTS_DEBUG_TIMING = os.getenv("TTS_DEBUG_TIMING", "0") == "1"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
REF_AUDIO_PATH = "sandy_ref.wav"
SPEAKER_INDEX = int(os.getenv("SPEAKER_INDEX")) if os.getenv("SPEAKER_INDEX") else None

SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
BLOCKSIZE   = int(os.getenv("TTS_BLOCKSIZE", "2048"))

TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "220"))
MIN_CHARS = int(os.getenv("TTS_MIN_CHARS", "120"))

EMPTY_CACHE_EACH_CHUNK = os.getenv("TTS_EMPTY_CACHE_EACH_CHUNK", "0") == "1"

LATENTS_PATH = os.getenv("TTS_LATENTS_PATH", "sandy_latents.pt")
FORCE_REBUILD_LATENTS = os.getenv("TTS_LATENTS_FORCE_REBUILD", "0") == "1"


def _cuda_is_usable():
    """Check if CUDA is available AND actually works for computation.
    Catches cases where the GPU architecture (e.g. sm_120/Blackwell)
    is not supported by the installed PyTorch build."""
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, device="cuda")
        _ = (x + 1).item()
        return True
    except Exception as e:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   ⚠️ CUDA detectado ({gpu_name}) pero no funcional para compute:")
        print(f"      {e}")
        print(f"   ↳ Usando CPU como fallback. Para GPU, instala PyTorch con cu128:")
        print(f"     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")
        return False


def check_vram():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   🔍 VRAM: {used:.2f}GB / {total:.2f}GB")


def normalize_tts_text(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""

    t = t.replace("¿", "").replace("¡", "")

    t = re.sub(r"^\s*(Sandy:|Asistente:)\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?im)^\s*(usuario|sandy|asistente)\s*:\s*", "", t)

    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"(?m)^\s*[\-\*\•]\s+", "", t)
    t = t.replace("*", " ").replace("_", " ")

    t = re.sub(r"https?://\S+", "un enlace", t)

    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    t = t.replace("...", ".")
    t = t.replace(":", ".")
    t = t.replace(";", ".")

    t = t.replace("E=mc²", "E igual a m por c al cuadrado")
    t = t.replace("mc²", "m por c al cuadrado")
    t = t.replace("²", " al cuadrado ")
    t = t.replace("³", " al cubo ")
    t = t.replace("=", " igual a ")

    t = re.sub(r"[\u200b-\u200f\uFEFF]", "", t)

    t = re.sub(r"\bDr\.", "Doctor", t)
    t = re.sub(r"\bSr\.", "Señor", t)
    t = re.sub(r"\bSra\.", "Señora", t)
    t = re.sub(r"\bDra\.", "Doctora", t)

    t = re.sub(r"\b[A-ZÁÉÍÓÚÑÜ]{4,}\b", lambda m: m.group(0).lower(), t)

    acr = {
        "CPU": "ce pe u",
        "GPU": "ge pe u",
        "RAM": "ram",
        "SSD": "ese ese de",
        "HTTP": "hache te te pe",
        "HTTPS": "hache te te pe ese",
        "API": "a pe i",
        "SQL": "ese cu ele",
        "URL": "u erre ele",
        "HTML": "hache te eme ele",
        "CSS": "ce ese ese",
        "JSON": "jeison",
        "XML": "equis eme ele",
        "PC": "pe ce",
    }
    for k, v in acr.items():
        t = re.sub(rf"\b{k}\b", v, t, flags=re.IGNORECASE)

    t = re.sub(r"\b(\d+)\.\s*", r"Paso \1. ", t)

    t = re.sub(r"[\r\n]+", ". ", t)

    t = re.sub(r"[^\w\s\.\,\!\?\-\(\)áéíóúñÁÉÍÓÚÑüÜ%]", "", t)

    t = re.sub(
        r"\b(?:[A-Za-zÁÉÍÓÚÑÜ]\s+){2,}[A-Za-zÁÉÍÓÚÑÜ]\b",
        lambda m: m.group(0).replace(" ", ""),
        t
    )

    t = re.sub(r"\s+", " ", t).strip()

    if t and t[-1] not in ".!?":
        t += "."

    return t


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def pack_chunks(sentences: list[str], min_chars: int, max_chars: int) -> list[str]:
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
            buf = ""

    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue

        candidate = (buf + " " + s).strip() if buf else s

        if len(candidate) > max_chars:
            if buf:
                flush()
                buf = s
            else:
                remaining = s
                while len(remaining) > max_chars:
                    cut = remaining[:max_chars]
                    if " " in cut:
                        cut = cut.rsplit(" ", 1)[0]
                    cut = cut.strip()
                    if cut:
                        chunks.append(cut)
                    remaining = remaining[len(cut):].strip() if cut else remaining[max_chars:].strip()
                buf = remaining
        else:
            buf = candidate

        if len(buf) >= min_chars and re.search(r"[.!?,]$", buf):
            flush()
        elif len(buf) >= max_chars - 5:
            flush()

    flush()

    merged: list[str] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        if len(ch) < 20 and merged:
            merged[-1] = (merged[-1].rstrip() + " " + ch).strip()
        else:
            merged.append(ch)

    return merged


def _fade_in_out(audio: np.ndarray, sr: int, ms: int = 6) -> np.ndarray:
    if audio.size == 0:
        return audio
    n = int(sr * (ms / 1000.0))
    n = min(n, audio.size // 2)
    if n <= 0:
        return audio

    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)

    audio = audio.astype(np.float32, copy=False)
    audio[:n] *= fade_in
    audio[-n:] *= fade_out
    return audio


class TTSService:
    def __init__(self):
        device = "cuda" if _cuda_is_usable() else "cpu"
        self._device = device
        print(f"🔊 Configurando Voz en {device.upper()}... (ID: {SPEAKER_INDEX})")

        os.environ["COQUI_TOS_AGREED"] = "1"

        if not os.path.exists(REF_AUDIO_PATH):
            raise FileNotFoundError(f"Falta '{REF_AUDIO_PATH}'")

        self.tts = TTS(MODEL_NAME).to(device)
        check_vram()
        print(f"✅ TTS cargado en {device.upper()}.")

        self._xtts_model = None
        try:
            self._xtts_model = self.tts.synthesizer.tts_model
        except Exception:
            self._xtts_model = None

        self._gpt_cond_latent = None
        self._speaker_embedding = None
        self._prepare_speaker_cache()
        self._move_latents_to_model_device()

        self.generation_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self._shutdown_event = threading.Event()

        # ✅ NUEVO: interrupción (barge-in)
        self._interrupt_event = threading.Event()

        self.first_audio_ready = threading.Event()

        self._warmup_model()

        self._reset_prebuffer = False
        self.generator_thread = threading.Thread(target=self._audio_generator, daemon=True)
        self.engine_thread = threading.Thread(target=self._audio_engine, daemon=True)
        self.generator_thread.start()
        self.engine_thread.start()

    def _drain_queue_safely(self, q: queue.Queue) -> int:
        """
        Vacía una Queue SIN romper unfinished_tasks (evita deadlocks en join()).
        Devuelve cuántos items drenó.
        """
        drained = 0
        while True:
            try:
                _ = q.get_nowait()
            except pyqueue.Empty:
                break
            else:
                drained += 1
                try:
                    q.task_done()
                except Exception:
                    pass
        return drained

    # ---------- Persistencia latents ----------
    def _ref_mtime(self) -> float:
        return os.path.getmtime(REF_AUDIO_PATH)

    def _load_latents_from_disk(self) -> bool:
        if FORCE_REBUILD_LATENTS:
            return False
        if not LATENTS_PATH or not os.path.exists(LATENTS_PATH):
            return False

        try:
            payload = torch.load(LATENTS_PATH, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                return False

            cached_mtime = payload.get("ref_mtime")
            if cached_mtime is None:
                return False
            if abs(float(cached_mtime) - float(self._ref_mtime())) > 1e-6:
                print("   ↳ [Latents] Cache inválido: cambió sandy_ref.wav. Recalculando...")
                return False

            if payload.get("model_name") != MODEL_NAME:
                print("   ↳ [Latents] Cache inválido: cambió MODEL_NAME. Recalculando...")
                return False

            if str(payload.get("torch_version")) != str(torch.__version__):
                print("   ↳ [Latents] Cache inválido: cambió versión de PyTorch. Recalculando...")
                return False

            if payload.get("sample_rate") != SAMPLE_RATE:
                print("   ↳ [Latents] Cache inválido: cambió SAMPLE_RATE. Recalculando...")
                return False

            gpt = payload.get("gpt_cond_latent")
            spk = payload.get("speaker_embedding")
            if gpt is None or spk is None:
                return False

            self._gpt_cond_latent = gpt
            self._speaker_embedding = spk
            print(f"   ✅ [Latents] Cargados desde disco: {LATENTS_PATH}")
            return True

        except Exception as e:
            print(f"   ⚠️ [Latents] No se pudo cargar cache: {e}")
            return False

    def _save_latents_to_disk(self) -> None:
        if not LATENTS_PATH:
            return
        try:
            gpt = self._gpt_cond_latent.detach().cpu() if hasattr(self._gpt_cond_latent, "detach") else self._gpt_cond_latent
            spk = self._speaker_embedding.detach().cpu() if hasattr(self._speaker_embedding, "detach") else self._speaker_embedding

            payload = {
                "ref_mtime": float(self._ref_mtime()),
                "gpt_cond_latent": gpt,
                "speaker_embedding": spk,
                "torch_version": str(torch.__version__),
                "model_name": str(MODEL_NAME),
                "sample_rate": int(SAMPLE_RATE),
            }

            torch.save(payload, LATENTS_PATH)
            print(f"   ✅ [Latents] Guardados en disco: {LATENTS_PATH}")
        except Exception as e:
            print(f"   ⚠️ [Latents] No se pudo guardar cache: {e}")

    def _prepare_speaker_cache(self):
        if self._xtts_model is None:
            return

        if self._load_latents_from_disk():
            return

        try:
            if hasattr(self._xtts_model, "get_conditioning_latents"):
                gpt_latent, spk_emb = self._xtts_model.get_conditioning_latents(audio_path=REF_AUDIO_PATH)
                self._gpt_cond_latent = gpt_latent
                self._speaker_embedding = spk_emb
                print("   ✅ [Latents] Calculados (XTTS). Guardando cache...")
                self._save_latents_to_disk()
        except Exception as e:
            print(f"   ⚠️ No se pudo calcular speaker latents: {e}")

    # ---------- Core TTS ----------
    def _warmup_model(self):
        print("   ↳ [Warmup] Preparando motor TTS...")
        try:
            if self._device == "cuda":
                torch.zeros((1,), device="cuda")
                torch.cuda.synchronize()

            _ = self._synthesize("Hola.")
            if self._device == "cuda":
                torch.cuda.synchronize()

            _ = self._synthesize("Probando.")
            if self._device == "cuda":
                torch.cuda.synchronize()

            print("   ✅ [Warmup] Listo.")
        except Exception as e:
            print(f"   ⚠️ [Warmup] Error no crítico: {e}")

    def _synthesize(self, text_item: str) -> np.ndarray:
        text_item = normalize_tts_text(text_item)
        if len(text_item) < 2:
            return np.zeros((1, 1), dtype=np.float32)

        t0 = time.perf_counter()
        used_path = "unknown"

        if (
            self._xtts_model is not None
            and self._gpt_cond_latent is not None
            and self._speaker_embedding is not None
            and hasattr(self._xtts_model, "inference")
        ):
            used_path = "inference"
            if TTS_DEBUG_TIMING:
                dev, dtype = self._model_device_dtype()
                print(
                    f"   🔎 PATH=inference | len={len(text_item)} "
                    f"| lat_dev={getattr(self._gpt_cond_latent,'device',None)} "
                    f"| lat_dtype={getattr(self._gpt_cond_latent,'dtype',None)} "
                    f"| model_dev={dev} | model_dtype={dtype}"
                )

            with torch.inference_mode():
                out = self._xtts_model.inference(
                    text=text_item,
                    language="es",
                    gpt_cond_latent=self._gpt_cond_latent,
                    speaker_embedding=self._speaker_embedding,
                )
            wav = out["wav"] if isinstance(out, dict) and "wav" in out else out

        else:
            used_path = "fallback_tts"
            if TTS_DEBUG_TIMING:
                print(f"   🔎 PATH=fallback_tts | len={len(text_item)}")

            kwargs = dict(
                text=text_item,
                speaker_wav=REF_AUDIO_PATH,
                language="es",
                split_sentences=False,
            )
            try:
                wav = self.tts.tts(**kwargs, speed=TTS_SPEED)
            except TypeError:
                wav = self.tts.tts(**kwargs)

        if TTS_DEBUG_TIMING:
            dt = (time.perf_counter() - t0) * 1000
            print(f"   ⏱️ TTS chunk: {dt:.0f} ms | path={used_path} | {text_item[:60]}")

        audio = np.asarray(wav, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        audio = _fade_in_out(audio, SAMPLE_RATE, ms=6)

        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        elif audio.ndim == 2 and audio.shape[1] != 1:
            audio = audio[:, :1]

        if self._device == "cuda" and EMPTY_CACHE_EACH_CHUNK:
            torch.cuda.empty_cache()

        return audio

    def interrupt(self):
        """
        Corta lo que se esté reproduciendo ahora mismo y limpia colas SIN deadlocks.
        """
        # Señal a motor de audio para que corte en el próximo bloque
        self._interrupt_event.set()

        # reset prebuffer
        self._reset_prebuffer = True
        self.first_audio_ready.clear()

        # ✅ Drenar colas de forma segura (NO usar q.queue.clear())
        try:
            self._drain_queue_safely(self.generation_queue)
            self._drain_queue_safely(self.audio_queue)
        except Exception:
            pass


    def _audio_generator(self):
        print("   ↳ [Generador Audio] 🟢 ONLINE")
        chunk_count = 0

        while not self._shutdown_event.is_set():

            if getattr(self, "_reset_prebuffer", False):
                chunk_count = 0
                self._reset_prebuffer = False
                self.first_audio_ready.clear()

            text_item = self.generation_queue.get()

            if text_item is None:
                self.generation_queue.task_done()
                break

            try:
                text_item = (text_item or "").strip()
                if len(text_item) < 2:
                    continue

                audio_data = self._synthesize(text_item)
                self.audio_queue.put(audio_data)

                chunk_count += 1
                if chunk_count == 1:
                    self.first_audio_ready.set()

            except Exception as e:
                print(f"❌ Error generando audio: {e}")

            finally:
                self.generation_queue.task_done()

    def _write_chunk_interruptible(self, stream: sd.OutputStream, audio_chunk: np.ndarray):
        """
        ✅ Escribe en bloques para poder cortar (barge-in) sin esperar el chunk completo.
        """
        if audio_chunk is None or audio_chunk.size == 0:
            return

        # Asegurar shape (N,1)
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk.reshape(-1, 1)

        n = audio_chunk.shape[0]
        i = 0

        # limpiar flag si ya venía de antes
        if self._interrupt_event.is_set():
            # NO lo limpiamos aquí: lo limpia quien interrumpe (o al cortar). Lo usamos como señal.
            pass

        while i < n and not self._shutdown_event.is_set():
            if self._interrupt_event.is_set():
                # cortar inmediatamente
                return

            end = min(i + BLOCKSIZE, n)
            stream.write(audio_chunk[i:end])
            i = end

    def _audio_engine(self):
        try:
            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                device=SPEAKER_INDEX,
                channels=1,
                dtype="float32",
                blocksize=BLOCKSIZE,
            ) as stream:
                print("   ↳ [Motor Audio] 🟢 ONLINE")

                while not self._shutdown_event.is_set():
                    audio_chunk = self.audio_queue.get()

                    if audio_chunk is None:
                        self.audio_queue.task_done()
                        break

                    try:
                        # si hubo interrupción pendiente antes de reproducir, consúmela
                        if self._interrupt_event.is_set():
                            # aborta buffers internos y limpia flag
                            try:
                                stream.abort()
                                stream.start()
                            except Exception:
                                pass
                            self._interrupt_event.clear()

                        # reproduce en bloques, cortable
                        self._write_chunk_interruptible(stream, audio_chunk)

                        # si se interrumpió durante la reproducción, aborta y limpia flag
                        if self._interrupt_event.is_set():
                            try:
                                stream.abort()
                                stream.start()
                            except Exception:
                                pass
                            self._interrupt_event.clear()

                    finally:
                        self.audio_queue.task_done()

        except Exception as e:
            print(f"❌ FALLO MOTOR DE AUDIO: {e}")

    def speak(self, text: str, overwrite: bool = False):
        text = (text or "").strip()
        if len(text) < 2:
            return

        if overwrite:
            # ✅ Si overwrite, también interrumpimos lo actual
            self.interrupt()

        text = normalize_tts_text(text)
        if len(text) < 2:
            return

        sentences = split_sentences(text)
        sentences = [s.strip() for s in sentences if s and s.strip()]
        if not sentences:
            return

        first_buf = sentences[0]
        i = 1

        while len(first_buf) < MIN_CHARS and i < len(sentences):
            candidate = (first_buf + " " + sentences[i]).strip()
            if len(candidate) > MAX_CHARS and len(first_buf) >= 20:
                break
            first_buf = candidate
            i += 1

        pending: list[str] = []
        if first_buf:
            pending.append(first_buf)

        rest = sentences[i:]
        chunks = pack_chunks(rest, min_chars=MIN_CHARS, max_chars=MAX_CHARS)

        for ch in chunks:
            ch = (ch or "").strip()
            if not ch:
                continue
            if len(ch) < 20 and pending:
                pending[-1] = (pending[-1].rstrip() + " " + ch).strip()
            else:
                pending.append(ch)

        final_pending: list[str] = []
        for item in pending:
            item = (item or "").strip()
            if not item:
                continue
            if final_pending and len(item) < 20:
                final_pending[-1] = (final_pending[-1].rstrip() + " " + item).strip()
            else:
                final_pending.append(item)

        for item in final_pending:
            item = (item or "").strip()
            if len(item) >= 2:
                self.generation_queue.put(item)

    def shutdown(self):
        self._shutdown_event.set()
        self.generation_queue.put(None)
        self.audio_queue.put(None)
        self.generator_thread.join(timeout=2)
        self.engine_thread.join(timeout=2)

    def _model_device_dtype(self):
        try:
            p = next(self._xtts_model.parameters())
            return p.device, p.dtype
        except Exception:
            dev = torch.device(self._device)
            return dev, torch.float16 if dev.type == "cuda" else torch.float32

    def _move_latents_to_model_device(self):
        if self._gpt_cond_latent is None or self._speaker_embedding is None:
            return

        dev, dtype = self._model_device_dtype()
        try:
            self._gpt_cond_latent = self._gpt_cond_latent.to(device=dev, dtype=dtype, non_blocking=True)
            self._speaker_embedding = self._speaker_embedding.to(device=dev, dtype=dtype, non_blocking=True)

            if hasattr(self._gpt_cond_latent, "contiguous"):
                self._gpt_cond_latent = self._gpt_cond_latent.contiguous()
            if hasattr(self._speaker_embedding, "contiguous"):
                self._speaker_embedding = self._speaker_embedding.contiguous()

            print(f"   ✅ [Latents] Movidos a {dev} y casteados a {dtype}")
        except Exception as e:
            print(f"   ⚠️ [Latents] No se pudieron mover/castear: {e}")
