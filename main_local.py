# main_local.py
from __future__ import annotations

import os
import re
import sys
import time
import traceback
import queue
import threading
import unicodedata
from collections import deque
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from app.services.audio.listening_service import ListeningService
from app.services.audio.tts_service import TTSService

from app.services import gemini as brain
from app.services.brain.intent_detector import IntentDetector

try:
    from app.services.brain.intent_logger import log_miss
except Exception:
    def log_miss(*args, **kwargs):
        return

load_dotenv()


# ==========================
# Env toggles
# ==========================
ENABLE_SPEAKER_ID = os.getenv("ENABLE_SPEAKER_ID", "0") == "1"
ENABLE_MULTI_PROFILES_ENV = os.getenv("ENABLE_MULTI_PROFILES", "0") == "1"

profiles = brain.profiles
query_gemini_stream = brain.query_gemini_stream
ENABLE_MULTI_PROFILES = bool(ENABLE_MULTI_PROFILES_ENV and profiles)

# ==========================
# Wake words / Commands
# ==========================
WAKE_WORDS = ["sandy", "sandi", "dandy", "oye sandy", "hey sandy"]

CONTINUE_TRIGGERS = (
    "continua",
    "continúa",
    "sigue",
    "retoma",
    "retomar",
    "continua lo que decias",
    "continúa lo que decías",
    "sandy continua",
    "sandy continúa",
)

VOICE_SAVE_TRIGGERS = (
    "guarda este tono",
    "guarda mi voz",
    "aprende mi voz",
    "aprende este tono",
    "memoriza mi voz",
    "registra este tono",
    "guarda este fragmento",
    "tambien guarda este tono",
    "también guarda este tono",
)

NONE_WORDS = ("ninguno", "ninguna", "no soy", "negativo", "para nada")

intent_detector = IntentDetector()

# ==========================
# Thresholds / timings
# ==========================
NEAR_STRANGER_MIN = float(os.getenv("NEAR_STRANGER_MIN", "0.20"))
NEAR_STRANGER_MAX = float(os.getenv("NEAR_STRANGER_MAX", "0.24"))
CONFIRM_TIMEOUT_SECONDS = float(os.getenv("CONFIRM_TIMEOUT_SECONDS", "5"))

# Barge-in gating (anti falsos positivos)
BARGEIN_MIN_CONF = float(os.getenv("BARGEIN_MIN_CONF", "0.18"))
BARGEIN_MIN_VOLUME = float(os.getenv("BARGEIN_MIN_VOLUME", "350"))
BARGEIN_MIN_DURATION = float(os.getenv("BARGEIN_MIN_DURATION", "0.35"))
POLL_INTERRUPT_EVERY = float(os.getenv("POLL_INTERRUPT_EVERY", "0.06"))

# Background buffer
BACKGROUND_MAX = int(os.getenv("BACKGROUND_MAX", "40"))
BG_MIN_TEXT_CHARS = int(os.getenv("BG_MIN_TEXT_CHARS", "10"))
BG_MIN_VOLUME = float(os.getenv("BG_MIN_VOLUME", "250"))  # filtra ruido muy suave
INCLUDE_BG_IN_CONTINUE_PROMPT = os.getenv("INCLUDE_BG_IN_CONTINUE_PROMPT", "1") == "1"

# (opcional) incluir background también en preguntas normales (por defecto apagado)
INCLUDE_BG_IN_NORMAL_PROMPT = os.getenv("INCLUDE_BG_IN_NORMAL_PROMPT", "0") == "1"

# ==========================
# Estado: pending + background
# ==========================
pending_responses: Dict[str, Dict[str, str]] = {}
background_buffer = deque(maxlen=BACKGROUND_MAX)


# ==========================
# Helpers
# ==========================
def _cleanup_sample_path(path: Optional[str]):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_none_choice(text: str) -> bool:
    t = _normalize_text(text)
    return any(w in t for w in NONE_WORDS)


def _is_affirmative(text: str) -> bool:
    t = _normalize_text(text)
    return any(x in t for x in ("si", "sí", "soy", "claro", "correcto", "afirmativo", "dale"))


def _is_negative(text: str) -> bool:
    t = _normalize_text(text)
    return any(x in t for x in ("no", "negativo", "para nada", "nunca"))


def _wants_save_voice(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in VOICE_SAVE_TRIGGERS)


def _wants_continue(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(k in t for k in CONTINUE_TRIGGERS)


def _speaker_key(speaker_id: Any) -> str:
    return str(speaker_id) if speaker_id else "global"


def _is_unknown_speaker_id(speaker_id: Any) -> bool:
    if speaker_id is None:
        return True
    sid = str(speaker_id).strip().lower()
    return sid in {"temp_id", "stranger"} or sid.startswith("temp_")


def is_addressed_to_sandy(text: str) -> Tuple[bool, str]:
    if not isinstance(text, str):
        return False, ""
    text_lower = text.lower().strip()
    for wake in WAKE_WORDS:
        if wake in text_lower:
            cleaned = text
            cleaned = cleaned.replace(wake, "", 1)
            cleaned = cleaned.replace(wake.capitalize(), "", 1)
            cleaned = cleaned.strip(" ,.:;!?¿¡\t\n\r")
            return True, cleaned if cleaned else text
    return False, text


def _get_fresh_turn(listener: ContinuousListener, after_ts: float, timeout: float):
    """
    Devuelve (text, sid, meta) del primer evento cuyo meta.captured_at >= after_ts.
    Descarta eventos viejos.
    """
    t_end = time.time() + float(timeout)
    while time.time() < t_end:
        item = listener.get(timeout=0.15)
        if not item:
            continue
        txt, sid, meta = _parse_result(item)
        cap = meta.get("captured_at")
        if isinstance(cap, (int, float)) and float(cap) >= float(after_ts):
            return txt, sid, meta
        # viejo -> descartar y limpiar wav si existe
        _cleanup_sample_path(meta.get("audio_path"))
    return "", None, {}


def _parse_result(res: Any) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Normaliza salida de ear.listen_and_transcribe(..., return_meta=True)
    """
    if isinstance(res, tuple) and len(res) == 3:
        user_text, speaker_id, meta = res
        meta = meta or {}
        return user_text or "", speaker_id, meta
    if isinstance(res, tuple) and len(res) == 2:
        user_text, speaker_id = res
        return user_text or "", speaker_id, {}
    return (res or ""), None, {}


def _store_background(text: str, speaker_id: Any, meta: Dict[str, Any], addressed: bool = False):
    """
    Guarda conversación de fondo (incluye lo que se oye mientras Sandy habla).
    Filtra ruido: requiere algo de volumen y texto mínimo.
    """
    t = (text or "").strip()
    if not t:
        return
    if len(t) < BG_MIN_TEXT_CHARS:
        return

    max_vol = meta.get("max_volume")
    if isinstance(max_vol, (int, float)) and float(max_vol) < BG_MIN_VOLUME:
        return

    background_buffer.append({
        "ts": time.time(),
        "who": str(speaker_id) if speaker_id else "?",
        "addressed": bool(addressed),
        "text": t,
    })


def _get_background_context(max_items: int = 10) -> str:
    if not background_buffer:
        return ""
    items = list(background_buffer)[-max_items:]
    lines = []
    for it in items:
        flag = "->Sandy" if it.get("addressed") else "BG"
        lines.append(f"- [{flag}] ({it.get('who')}) {it.get('text')}")
    return "\n".join(lines)


def _save_pending(speaker_id: Any, original_query: str, partial_response: list[str]):
    key = _speaker_key(speaker_id)
    text = " ".join([x.strip() for x in partial_response if x and x.strip()]).strip()
    if not text:
        return
    pending_responses[key] = {
        "original_query": (original_query or "").strip(),
        "partial_response": text,
    }


def _peek_pending(speaker_id: Any) -> Optional[Dict[str, str]]:
    return pending_responses.get(_speaker_key(speaker_id))


def _pop_pending(speaker_id: Any) -> Optional[Dict[str, str]]:
    return pending_responses.pop(_speaker_key(speaker_id), None)


def _build_continue_prompt(pending: Dict[str, str], bg: str) -> str:
    original = pending.get("original_query", "")
    partial = pending.get("partial_response", "")

    prompt = (
        "Te interrumpieron mientras respondías. Retoma la respuesta con naturalidad.\n"
        "Reglas:\n"
        "1) No repitas desde cero, continúa desde lo ya dicho.\n"
        "2) Si el usuario cambió de tema, reconoce la interrupción y decide si retomas o no.\n"
        "3) Si decides NO retomar, explica por qué en una frase.\n\n"
        f"Pregunta original del usuario:\n{original}\n\n"
        f"Respuesta parcial que ya ibas diciendo:\n{partial}\n"
    )
    if bg:
        prompt += f"\nContexto reciente de conversación de fondo:\n{bg}\n"
    return prompt


def _tts_busy(mouth: TTSService) -> bool:
    """
    Mucho más fiable que .empty(): unfinished_tasks cubre "reproduciendo ahora".
    """
    try:
        return (mouth.generation_queue.unfinished_tasks > 0) or (mouth.audio_queue.unfinished_tasks > 0)
    except Exception:
        # fallback suave
        return (not mouth.generation_queue.empty()) or (not mouth.audio_queue.empty())


def _should_barge_in(user_text: str, speaker_id: Any, meta: Dict[str, Any]) -> bool:
    """
    Gating fuerte anti falsos positivos.
    Requiere wake word + volumen + duración.
    Si speaker desconocido, exige confidence >= BARGEIN_MIN_CONF.
    """
    is_for_me, _ = is_addressed_to_sandy(user_text)
    if not is_for_me:
        return False

    conf = meta.get("confidence")
    max_vol = meta.get("max_volume")
    dur = meta.get("duration_s")

    if max_vol is None or dur is None:
        return False

    if float(max_vol) < BARGEIN_MIN_VOLUME:
        return False
    if float(dur) < BARGEIN_MIN_DURATION:
        return False

    known = not _is_unknown_speaker_id(speaker_id)
    if known:
        return True

    # desconocido: más estricto
    if isinstance(conf, (int, float)) and float(conf) >= BARGEIN_MIN_CONF:
        return True

    return False


def _extract_declared_name(intent_detector: IntentDetector, text: str) -> Optional[str]:
    if not text:
        return None
    name = intent_detector.detect_self_introduction(text)
    if name:
        return name.strip()

    m = re.search(
        r"\b(?:no\s*,?\s*)?(?:yo\s+)?soy\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'\-]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'\-]+)?)\b",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def _name_in_candidates(spk_service: Any, name: str, candidates: list[dict]) -> bool:
    if not name or not candidates:
        return False
    n = spk_service.normalize_person_token(name)
    if not n:
        return False
    for c in candidates:
        cn = c.get("name", "")
        if cn and spk_service.normalize_person_token(cn) == n:
            return True
    return False


def _pick_candidate_from_text(spk_service: Any, text: str, candidates: list[dict]) -> Optional[dict]:
    if not text or not candidates:
        return None
    norm = spk_service.normalize_person_token(text)
    if not norm:
        return None

    for c in candidates:
        name = c.get("name", "")
        if not name:
            continue
        name_norm = spk_service.normalize_person_token(name)
        first = name_norm.split(" ")[0] if name_norm else ""
        if name_norm and name_norm in norm:
            return c
        if first and f" {first} " in f" {norm} ":
            return c
    return None


def _format_candidates_for_tts(candidates: list[dict], max_names: int = 2) -> list[str]:
    out = []
    for c in candidates[:max_names]:
        n = c.get("name")
        if n:
            out.append(str(n))
    return out


# ==========================
# Continuous Listener Thread
# ==========================
class ContinuousListener:
    """
    Escucha continua en background.
    """
    def __init__(self, ear: ListeningService):
        self.ear = ear
        self.q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                res = self.ear.listen_and_transcribe(allow_unknown=True, return_meta=True)
                self.q.put(res)
            except Exception as e:
                print(f"⚠️ [Listener] Error: {e}")
                time.sleep(0.2)

    def drain(self, max_items: int = 200):
        """Vacía la cola para evitar leer eventos viejos en confirmaciones."""
        n = 0
        while n < max_items:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
            else:
                n += 1

    def get(self, timeout: float = 0.1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def put_back(self, item: Any):
        try:
            self.q.put(item)
        except Exception:
            pass

    def stop(self):
        self.stop_event.set()


# ==========================
# Core: hablar con barge-in + background capture
# ==========================
def _run_gemini_and_speak(
    prompt: str,
    speaker_id: Any,
    listener: ContinuousListener,
    mouth: TTSService,
    original_query_for_pending: str,
) -> Tuple[bool, list[str]]:
    """
    Devuelve: (interrupted, full_response_sentences)
    - Si se interrumpe, guarda pending y re-inserta el evento de interrupción en la cola del listener.
    - Mientras habla, captura background (no dirigido a Sandy) y también "dirigido pero inválido" como logs.
    """
    first_chunk = True
    full_response: list[str] = []
    interrupted = False

    bg_in_prompt = ""
    if INCLUDE_BG_IN_NORMAL_PROMPT:
        bg_in_prompt = _get_background_context(max_items=10)
        if bg_in_prompt:
            prompt = f"{prompt}\n\nContexto de fondo (no necesariamente dirigido a ti):\n{bg_in_prompt}\n"

    gen = (
        query_gemini_stream(prompt, speaker_id)
        if (ENABLE_MULTI_PROFILES and speaker_id)
        else query_gemini_stream(prompt)
    )

    try:
        for sentence in gen:
            clean = re.sub(r"^\s*Sandy:\s*", "", sentence or "", flags=re.IGNORECASE).strip()
            if not clean:
                continue

            full_response.append(clean)
            mouth.speak(clean, overwrite=first_chunk)

            # prebuffer
            if first_chunk and hasattr(mouth, "first_audio_ready"):
                try:
                    mouth.first_audio_ready.wait(timeout=1.2)
                except Exception:
                    pass
            first_chunk = False

            # Mientras el TTS está "ocupado", vigilamos interrupciones y guardamos background
            while _tts_busy(mouth):
                intr = listener.get(timeout=POLL_INTERRUPT_EVERY)
                if not intr:
                    continue

                intr_text, intr_sid, intr_meta = _parse_result(intr)
                if not intr_text:
                    _cleanup_sample_path(intr_meta.get("audio_path"))
                    continue

                intr_is_for_me, _ = is_addressed_to_sandy(intr_text)

                if _should_barge_in(intr_text, intr_sid, intr_meta):
                    print("🛑 [BARGE-IN] Interrupción válida detectada")
                    # cortar TTS
                    try:
                        if hasattr(mouth, "interrupt"):
                            mouth.interrupt()
                        else:
                            # fallback (menos limpio)
                            mouth.speak("", overwrite=True)
                    except Exception:
                        pass

                    # guardar estado parcial
                    _save_pending(speaker_id, original_query_for_pending, full_response)

                    # reinsertar la interrupción para procesarla como próximo turno
                    listener.put_back(intr)

                    interrupted = True
                    break
                else:
                    # no era interrupción válida -> lo tratamos como background (incluye intentos suaves)
                    _store_background(intr_text, intr_sid, intr_meta, addressed=intr_is_for_me)
                    _cleanup_sample_path(intr_meta.get("audio_path"))

            if interrupted:
                break

    finally:
        try:
            gen.close()
        except Exception:
            pass

    if not interrupted:
        # esperar a que termine de hablar
        try:
            mouth.generation_queue.join()
            mouth.audio_queue.join()
        except Exception:
            pass

    return interrupted, full_response


# ==========================
# Main
# ==========================
def main():
    print("\n🚀 INICIANDO SANDY CORE...")
    print("👉 Ctrl+C para salir.\n")

    if not os.path.exists("sandy_ref.wav"):
        print("\n🛑 ALTO AHÍ: Falta 'sandy_ref.wav'.")
        return

    ear: Optional[ListeningService] = None
    mouth: Optional[TTSService] = None
    listener: Optional[ContinuousListener] = None

    try:
        print("--- CARGANDO SERVICIOS ---")
        ear = ListeningService()
        mouth = TTSService()

        if ENABLE_SPEAKER_ID:
            print("🎙️  [Speaker ID] Activo")

        if ENABLE_MULTI_PROFILES:
            print("👥 [Multi-User + Social Logic] Activo")
        else:
            if ENABLE_MULTI_PROFILES_ENV and not profiles:
                print("⚠️ ENABLE_MULTI_PROFILES=1 pero profiles no inicializó. Desactivando.")
            else:
                print("👤 [Multi-User] Desactivado")

        print("--------------------------\n")

        # ENROLLMENT si no hay speakers.pkl (antes de arrancar escucha continua para no pelear por el micro)
        if (
            ENABLE_SPEAKER_ID
            and ear.speaker_service is not None
            and getattr(ear.speaker_service, "recognizer", None) is not None
        ):
            if not os.path.exists("sandy_voices_db/speakers.pkl"):
                print("⚠️ PRIMERA VEZ: Necesito registrar tu voz")
                name = input("¿Cómo te llamas? ").strip()
                success = ear.register_primary_user(name)

                if success and ENABLE_MULTI_PROFILES and profiles and ear.primary_user_id:
                    profiles.set_current_speaker(ear.primary_user_id)
                    profiles.set_primary_name(ear.primary_user_id, name)
                    profiles._save_profile(ear.primary_user_id)
                print()

        print("✅ Sandy está lista. Di 'Sandy' para hablarme.")

        # ✅ Arrancar escucha continua
        listener = ContinuousListener(ear)

        while True:
            res = listener.get(timeout=0.5)
            if not res:
                continue

            user_text, speaker_id, meta = _parse_result(res)
            audio_path = meta.get("audio_path")

            if not user_text:
                _cleanup_sample_path(audio_path)
                continue

            # Display name
            if speaker_id and ENABLE_MULTI_PROFILES and profiles:
                profiles.set_current_speaker(speaker_id)
                user_name = profiles.get_display_name(speaker_id)
                print(f"📝 {user_name}: {user_text}")
            else:
                print(f"📝 Tú: {user_text}")

            # Wake word
            is_for_me, cleaned_text = is_addressed_to_sandy(user_text)

            # Si NO era para Sandy -> background buffer y seguimos
            if not is_for_me:
                _store_background(user_text, speaker_id, meta, addressed=False)
                _cleanup_sample_path(audio_path)
                continue

            # Salir
            if "salir" in cleaned_text.lower() or "apágate" in cleaned_text.lower():
                print("🛑 Apagando...")
                mouth.speak("Adiós.", overwrite=True)
                mouth.generation_queue.join()
                mouth.audio_queue.join()
                _cleanup_sample_path(audio_path)
                break

            # ✅ COMANDO: CONTINÚA
            if _wants_continue(cleaned_text):
                pending = _peek_pending(speaker_id)
                if not pending:
                    mouth.speak("No tengo nada pendiente que continuar.", overwrite=True)
                    mouth.generation_queue.join()
                    mouth.audio_queue.join()
                    _cleanup_sample_path(audio_path)
                    continue

                bg = _get_background_context(max_items=10) if INCLUDE_BG_IN_CONTINUE_PROMPT else ""
                continue_prompt = _build_continue_prompt(pending, bg)

                print("🧠 [Continue] Retomando respuesta interrumpida...")
                interrupted, full_resp = _run_gemini_and_speak(
                    prompt=continue_prompt,
                    speaker_id=speaker_id,
                    listener=listener,
                    mouth=mouth,
                    original_query_for_pending=pending.get("original_query", "continuar"),
                )

                if not interrupted:
                    _pop_pending(speaker_id)
                    print(f"🤖 Sandy: {' '.join(full_resp)}")

                _cleanup_sample_path(audio_path)
                continue

            # ====== COMANDO MANUAL: GUARDAR TONO ======
            if ENABLE_SPEAKER_ID and ear.speaker_service and _wants_save_voice(cleaned_text):
                target_id = speaker_id if not _is_unknown_speaker_id(speaker_id) else ear.primary_user_id

                if not audio_path or not os.path.exists(audio_path):
                    mouth.speak("No tengo el audio de este turno para guardarlo.", overwrite=True)
                    mouth.generation_queue.join()
                    mouth.audio_queue.join()
                    continue

                if not target_id:
                    mouth.speak("No sé a quién asignarlo. Di tu nombre y repite.", overwrite=True)
                    mouth.generation_queue.join()
                    mouth.audio_queue.join()
                    _cleanup_sample_path(audio_path)
                    continue

                ok = bool(ear.speaker_service.add_embedding_to_speaker(str(target_id), audio_path))
                _cleanup_sample_path(audio_path)

                mouth.speak(
                    "Listo. Guardé esta muestra de tu voz." if ok else "No la guardé. Puede ser duplicada o ya no hay cupo.",
                    overwrite=True
                )
                mouth.generation_queue.join()
                mouth.audio_queue.join()
                continue

            # ====== stranger “cercano” -> confirmación A/B/ninguno o registrar nuevo ======
            conf = meta.get("confidence")
            candidates = meta.get("candidates") or []

            if (
                ENABLE_SPEAKER_ID
                and ear.speaker_service
                and _is_unknown_speaker_id(speaker_id)
                and isinstance(conf, (float, int))
                and audio_path and os.path.exists(audio_path)
                and candidates
            ):
                candidates = sorted(candidates, key=lambda x: float(x.get("score", -1.0)), reverse=True)
                top1 = float(candidates[0].get("score", -1.0))
                should_ask = (NEAR_STRANGER_MIN <= float(conf) <= NEAR_STRANGER_MAX) or (NEAR_STRANGER_MIN <= top1 <= NEAR_STRANGER_MAX)

                if should_ask:
                    top_names = _format_candidates_for_tts(candidates, max_names=2)
                    prompt = (
                        f"¿Eres {top_names[0]} o {top_names[1]}? di el nombre o di ninguno."
                        if len(top_names) >= 2 else f"¿Eres {top_names[0]}? di sí o no."
                    )

                    # antes de hablar, limpia eventos viejos
                    listener.drain()

                    mouth.speak(prompt, overwrite=True)
                    mouth.generation_queue.join()
                    mouth.audio_queue.join()

                    ask_ts = time.time()  # marca fin de pregunta (aprox)

                    confirm_text, _, confirm_meta = _get_fresh_turn(
                        listener=listener,
                        after_ts=ask_ts,
                        timeout=CONFIRM_TIMEOUT_SECONDS
                    )
                    print(f"🧪 [Voz] Confirmación oída: {confirm_text!r}")

                    if _is_none_choice(confirm_text):
                        _cleanup_sample_path(audio_path)
                        _cleanup_sample_path(confirm_meta.get("audio_path"))
                        mouth.speak("Ok. No guardaré esa muestra.", overwrite=True)
                        mouth.generation_queue.join()
                        mouth.audio_queue.join()
                        continue

                    chosen = None
                    if len(top_names) >= 2:
                        chosen = _pick_candidate_from_text(ear.speaker_service, confirm_text, candidates)
                    else:
                        if _is_affirmative(confirm_text) and not _is_negative(confirm_text):
                            chosen = candidates[0] if candidates else None

                    if chosen:
                        ok = bool(ear.speaker_service.add_embedding_to_speaker(str(chosen["speaker_id"]), audio_path))
                        _cleanup_sample_path(audio_path)
                        _cleanup_sample_path(confirm_meta.get("audio_path"))
                        mouth.speak(
                            f"Perfecto. Guardé esta muestra como {chosen['name']}." if ok else "No pude guardarla, quizá era duplicada o ya no hay cupo.",
                            overwrite=True
                        )
                        mouth.generation_queue.join()
                        mouth.audio_queue.join()
                        continue

                    declared_name = _extract_declared_name(intent_detector, confirm_text)
                    if declared_name and not _name_in_candidates(ear.speaker_service, declared_name, candidates):
                        mouth.speak(f"¿Confirmas que te registro como {declared_name}? di sí o no.", overwrite=True)
                        mouth.generation_queue.join()
                        mouth.audio_queue.join()

                        confirm2_res = listener.get(timeout=CONFIRM_TIMEOUT_SECONDS)
                        confirm2_text, _, confirm2_meta = _parse_result(confirm2_res) if confirm2_res else ("", None, {})
                        print(f"🧪 [Voz] Confirmación final: {confirm2_text!r}")

                        if _is_affirmative(confirm2_text) and not _is_negative(confirm2_text):
                            new_id = ear.speaker_service.register_speaker(audio_path, declared_name)
                            _cleanup_sample_path(audio_path)
                            _cleanup_sample_path(confirm_meta.get("audio_path"))
                            _cleanup_sample_path(confirm2_meta.get("audio_path"))

                            if new_id:
                                if ENABLE_MULTI_PROFILES and profiles:
                                    profiles.set_current_speaker(new_id)
                                    profiles.set_primary_name(new_id, declared_name)
                                    profiles._save_profile(new_id)
                                mouth.speak(f"Listo. Te registré como {declared_name}.", overwrite=True)
                            else:
                                mouth.speak("No pude registrarte. Hubo un error con tu voz.", overwrite=True)

                            mouth.generation_queue.join()
                            mouth.audio_queue.join()
                            continue

                        _cleanup_sample_path(audio_path)
                        _cleanup_sample_path(confirm_meta.get("audio_path"))
                        _cleanup_sample_path(confirm2_meta.get("audio_path"))
                        mouth.speak("Ok. No registraré esa voz.", overwrite=True)
                        mouth.generation_queue.join()
                        mouth.audio_queue.join()
                        continue

                    _cleanup_sample_path(audio_path)
                    _cleanup_sample_path(confirm_meta.get("audio_path"))
                    mouth.speak("Ok. No guardaré esa muestra.", overwrite=True)
                    mouth.generation_queue.join()
                    mouth.audio_queue.join()
                    continue

            # ====== PROCESAR GEMINI (con barge-in + pending + background) ======
            print("🧠 Procesando...")

            interrupted, full_resp = _run_gemini_and_speak(
                prompt=cleaned_text,
                speaker_id=speaker_id,
                listener=listener,
                mouth=mouth,
                original_query_for_pending=cleaned_text,
            )

            if not interrupted:
                print(f"🤖 Sandy: {' '.join(full_resp)}")
            else:
                # No respondemos completo porque quedó pending guardado
                print("⏸️ [Estado] Respuesta pausada. Puedes decir: 'Sandy continúa'.")

            _cleanup_sample_path(audio_path)

    except KeyboardInterrupt:
        print("\n\n🛑 Apagando Sandy...")

    except Exception as e:
        print("\n💥 ERROR:")
        print(str(e))
        traceback.print_exc()

    finally:
        try:
            if listener:
                listener.stop()
        except Exception:
            pass
        try:
            if mouth:
                mouth.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
