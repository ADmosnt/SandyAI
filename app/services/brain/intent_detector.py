# app/services/brain/intent_detector.py
import re
from typing import Optional, List, Tuple


class IntentDetector:
    """
    Detector heurístico determinista para:
    - nombre real ("me llamo", "mi nombre es", "soy")
    - apodo ("me dicen", "me llaman", "dime")
    - link ("me llamo X pero me dicen Y")
    - introducción de terceros ("te presento a X", "este es X")
    - menciones ("dile a X", "habla con X", "X dijo ...")
    """

    # Basura SOLO si aparece al INICIO o al FINAL del candidato
    INVALID_EDGE = {
        "un", "una", "el", "la", "los", "las", "mi", "tu", "su", "sus",
        "esto", "esta", "ese", "esa", "eso", "a", "de", "del", "que", "en", "con",
        "feliz", "triste", "cansado", "bien", "mal",
        "ingeniero", "programador", "estudiante", "persona", "amigo", "amiga",
        "doctor", "doctora", "problema", "cosa", "algo", "donde", "cuando",
        "hoy", "mañana", "ahora", "porfa", "porfavor",
    }

    # Palabras que NO deben aparecer dentro del nombre (cortan el candidato)
    INVALID_HARD = {
        # conectores / gramática conversacional
        "y", "pero", "porque", "para", "entonces",
        "que", "se", "si", "ya", "pues",

        # muletillas
        "oye", "mira", "bueno", "bro", "mano", "wey", "we",

        # temporales (cortan "Velok mañana")
        "hoy", "mañana", "ahora",

        # cortesía / ruido
        "porfa", "porfavor", "pls", "please", "por",

        # verbos/ruido común tras el nombre
        "necesito", "quiero", "quisiera", "puedes", "podrias",
        "registres", "registrar", "registre", "apure", "apurate", "apúrense",

        "eso", "esa", "este", "esta",

        "funciona", "sirve", "va",
    }

    BLOCKED_NAMES = {"sandy", "sandi"}

    # palabras reservadas que NO pueden ser parte del nombre
    _RESERVED = r"(?:soy|me|mi|dime|llamo|dicen|llaman|nombre)"

    # Letras válidas (incluye tildes y ü)
    _LETTERS = r"A-ZÁÉÍÓÚÑÜa-záéíóúñü"

    # token de nombre (evita tragarse keywords como "soy")
    NAME_TOKEN = rf"(?!{_RESERVED}\b)[{_LETTERS}]+(?:[-'][{_LETTERS}]+)?"

    # Captura 1 a 4 tokens (sin capturar internamente para no romper groups)
    _NAME_CAPTURE = rf"['\"\(]*{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}}['\"\)]*"

    # Anti-bug g4br13l => "g": (?!\w) evita prefix-match dentro de tokens alfanum
    _NAME_GROUP = rf"({ _NAME_CAPTURE })(?!\w)"

    # --- Patrones base ---
    _SELF_NAME_RE = re.compile(
        rf"(?:me\s+llamo|mi\s+nombre\s+es|soy)\s+{_NAME_GROUP}",
        re.IGNORECASE,
    )

    _NICK_RE = re.compile(
        rf"(?:me\s+dicen|me\s+llaman|dime)\s+{_NAME_GROUP}",
        re.IGNORECASE,
    )

    _ALIAS_LINK_RE = re.compile(
        rf"(?:me\s+llamo|mi\s+nombre\s+es|soy)\s+{_NAME_GROUP}\s*(?:,|\s)+"
        rf"(?:pero\s+)?(?:me\s+)?(?:dicen|llaman)\s+{_NAME_GROUP}",
        re.IGNORECASE,
    )

    _INTRO_RE_LIST = [
        re.compile(
            rf"(?:te\s+presento\s+a|conoce\s+a|saluda\s+a|les\s+presento\s+a)\s+"
            rf"(?:mi\s+(?:amigo|amiga|colega|hermano|hermana)\s+)?{_NAME_GROUP}",
            re.IGNORECASE,
        ),
        re.compile(rf"(?:este|esta)\s+es\s+{_NAME_GROUP}", re.IGNORECASE),
    ]

    _MENTION_RE_LIST = [
        re.compile(
            rf"(?:dile\s+a|habla\s+con|llama\s+a)\s+{_NAME_GROUP}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:te\s+presento\s+a|conoce\s+a|saluda\s+a)\s+"
            rf"(?:mi\s+(?:amigo|amiga|colega|hermano|hermana)\s+)?{_NAME_GROUP}",
            re.IGNORECASE,
        ),
        re.compile(rf"(?:este|esta)\s+es\s+{_NAME_GROUP}", re.IGNORECASE),
        re.compile(rf"(?:según|segun)\s+{_NAME_GROUP}", re.IGNORECASE),
        re.compile(rf"{_NAME_GROUP}\s+(?:dijo|dice|mencionó|menciono|cree)\b", re.IGNORECASE),
        re.compile(rf"(?:dijo|dice|mencionó|menciono|cree)\s+{_NAME_GROUP}", re.IGNORECASE),
    ]

    @staticmethod
    def _clean_and_validate(raw_name: str, is_soy_pattern: bool = False) -> Optional[str]:
        name = (raw_name or "").strip().strip(" \"'“”‘’.,;:!?()[]{}").strip()
        if not name:
            return None

        parts = name.split()

        # 1) Corte interno por INVALID_HARD
        for i, p in enumerate(parts):
            if p.lower() in IntentDetector.INVALID_HARD:
                parts = parts[:i]
                break

        # 2) Limpieza de extremos por INVALID_EDGE
        while parts and parts[-1].lower() in IntentDetector.INVALID_EDGE:
            parts.pop()
        while parts and parts[0].lower() in IntentDetector.INVALID_EDGE:
            parts.pop()

        if not parts or len(parts) > 4:
            return None

        final_name = " ".join(parts)
        low_name = final_name.lower()

        # 3) No dígitos
        if any(ch.isdigit() for ch in final_name):
            return None

        # 4) Bloqueo explícito (Sandy no se registra)
        if any(p in IntentDetector.BLOCKED_NAMES for p in low_name.split()):
            return None

        # 5) Regla dura para "soy X": si es 1 palabra y todo minúscula => None
        if is_soy_pattern:
            has_upper = any(c.isupper() for c in final_name)
            if len(parts) == 1 and not has_upper:
                return None

        return final_name

    @staticmethod
    def _check_negation(prefix: str) -> bool:
        """
        Si hay negación cerca del match ("no", "nunca", "jamás"),
        ignoramos ese match. Si hay puntuación fuerte justo antes,
        resetea la negación.
        """
        if re.search(r"[,.;!]\s*$", prefix or ""):
            return False
        words = re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]+", (prefix or "").lower())
        return any(w in {"no", "nunca", "jamás", "jamas"} for w in words[-3:])

    # --- API usada por tu main_local.py ---

    @staticmethod
    def detect_introduction(text: str) -> Optional[str]:
        """
        Detecta presentación de un tercero para enrollment:
        "Te presento a X", "Conoce a X", "Este es X"
        """
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return None

        for pat in IntentDetector._INTRO_RE_LIST:
            for m in pat.finditer(text):
                candidate = IntentDetector._clean_and_validate(m.group(1), is_soy_pattern=False)
                if candidate:
                    return candidate
        return None

    @staticmethod
    def detect_alias_link(text: str) -> Optional[Tuple[str, str]]:
        """
        Detecta: "me llamo Alan pero me dicen Lasher" => ("Alan", "Lasher")
        """
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return None

        for m in IntentDetector._ALIAS_LINK_RE.finditer(text):
            # group(1) = primer nombre, group(2) = segundo nombre
            real_raw = m.group(1)
            nick_raw = m.group(2)

            real = IntentDetector._clean_and_validate(real_raw, is_soy_pattern=m.group(0).lower().strip().startswith("soy"))
            nick = IntentDetector._clean_and_validate(nick_raw, is_soy_pattern=False)

            if real and nick:
                return (real, nick)

        return None

    @staticmethod
    def detect_self_introduction(text: str) -> Optional[str]:
        """
        SOLO nombre real: me llamo / mi nombre es / soy
        (Los apodos van por detect_nickname_introduction)
        """
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return None

        last_end = 0
        for m in IntentDetector._SELF_NAME_RE.finditer(text):
            local_prefix = text[last_end:m.start()]
            if IntentDetector._check_negation(local_prefix):
                last_end = m.end()
                continue

            is_soy = m.group(0).lower().strip().startswith("soy")
            candidate = IntentDetector._clean_and_validate(m.group(1), is_soy_pattern=is_soy)
            last_end = m.end()

            if candidate:
                return candidate

        return None

    @staticmethod
    def detect_nickname_introduction(text: str) -> Optional[str]:
        """
        Apodos/handles: "me dicen X", "me llaman X", "dime X"
        """
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return None

        last_end = 0
        for m in IntentDetector._NICK_RE.finditer(text):
            local_prefix = text[last_end:m.start()]
            if IntentDetector._check_negation(local_prefix):
                last_end = m.end()
                continue

            candidate = IntentDetector._clean_and_validate(m.group(1), is_soy_pattern=False)
            last_end = m.end()

            if candidate:
                return candidate

        return None

    @staticmethod
    def extract_mentioned_names(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return []

        found: List[str] = []
        seen = set()

        for pat in IntentDetector._MENTION_RE_LIST:
            for m in pat.finditer(text):
                candidate = IntentDetector._clean_and_validate(m.group(1), is_soy_pattern=False)
                if candidate:
                    key = candidate.lower()
                    if key not in seen:
                        seen.add(key)
                        found.append(candidate)

        return found
