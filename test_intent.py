# test_intent.py
# Ejecuta:
#   python test_intent.py
#
# Modo estricto (los XFAIL fallan):
#   STRICT=1 python test_intent.py

import os
from app.services.brain.intent_detector import IntentDetector

STRICT = os.getenv("STRICT", "0") == "1"


def _assert_equal(i: int, method: str, text: str, expected, got, xfail: bool, reason: str):
    ok = (got == expected)

    if ok and xfail:
        print(f"UPASS {i:03d} | {method:<12} | {text!r} -> {got!r} | (antes XFAIL) {reason}")
        return "upass"

    if ok:
        print(f"PASS  {i:03d} | {method:<12} | {text!r} -> {got!r}")
        return "pass"

    if xfail and not STRICT:
        print(f"XFAIL {i:03d} | {method:<12} | {text!r}")
        print(f"      Esperaba: {expected!r}")
        print(f"      Obtuve:   {got!r}")
        print(f"      Motivo:   {reason}")
        return "xfail"

    print(f"FAIL  {i:03d} | {method:<12} | {text!r}")
    print(f"      Esperaba: {expected!r}")
    print(f"      Obtuve:   {got!r}")
    if xfail:
        print(f"      (Era XFAIL, pero STRICT=1) Motivo: {reason}")
    raise SystemExit(1)


def run_tests():
    d = IntentDetector()

    # Helpers para soportar dos contratos:
    # - v2: detect_self_introduction() = nombre real
    #      detect_nickname_introduction() = apodo ("me dicen", "me llaman", "dime")
    # - v1: detect_self_introduction() también cubre apodos
    def _detect_real_name(text: str):
        return d.detect_self_introduction(text)

    def _detect_nickname(text: str):
        if hasattr(d, "detect_nickname_introduction"):
            return d.detect_nickname_introduction(text)
        # fallback legacy
        return d.detect_self_introduction(text)

    tests = [
        # =========================================================
        # SELF (NOMBRE REAL) - Básicos
        # =========================================================
        ("self_name", "Hola, me llamo Gabriel", "Gabriel", False, ""),
        ("self_name", "mi nombre es Alan Gabriel", "Alan Gabriel", False, ""),
        ("self_name", "Me llamo Juan Carlos", "Juan Carlos", False, ""),
        ("self_name", "Me llamo María de los Ángeles", "María de los Ángeles", False, ""),

        # =========================================================
        # SELF (APODO) - Básicos
        # =========================================================
        ("self_nick", "me dicen Velok", "Velok", False, ""),
        ("self_nick", "me llaman ElFenix", "ElFenix", False, ""),
        ("self_nick", "dime Gabo", "Gabo", False, ""),

        # =========================================================
        # "soy" (reglas)
        # =========================================================
        ("self_name", "soy gabriel", None, False, ""),                  # 1 palabra minúscula -> None
        ("self_name", "Soy Gabriel", "Gabriel", False, ""),             # 1 palabra con mayúscula -> OK
        ("self_name", "soy gabriel rondon", "gabriel rondon", False, ""),  # 2 palabras minúsculas -> OK
        ("self_name", "Soy Juan Carlos", "Juan Carlos", False, ""),

        # =========================================================
        # Ruido típico / formato
        # =========================================================
        ("self_name", "   me    llamo     Gabriel   ", "Gabriel", False, ""),
        ("self_name", "Me llamo Gabriel.", "Gabriel", False, ""),
        ("self_name", "Me llamo 'Gabriel'", "Gabriel", False, ""),
        ("self_name", 'Me llamo "Gabriel"', "Gabriel", False, ""),
        ("self_name", "Mi nombre es (Gabriel)", "Gabriel", False, ""),
        ("self_name", "eh me llamo Gabriel", "Gabriel", False, ""),
        ("self_name", "bro mi nombre es Gabriel", "Gabriel", False, ""),
        ("self_name", "yo mi nombre es Gabriel", "Gabriel", False, ""),
        ("self_name", "yo me llamo Gabriel", "Gabriel", False, ""),
        ("self_name", "me llamo Gabriel y soy ingeniero", "Gabriel", False, ""),

        # guión/apóstrofe
        ("self_name", "Me llamo Jean-Luc", "Jean-Luc", False, ""),
        ("self_name", "Me llamo O'Connor", "O'Connor", False, ""),

        # =========================================================
        # Trampas (None)
        # =========================================================
        ("self_name", "Soy un programador", None, False, ""),
        ("self_name", "Soy ingeniero", None, False, ""),
        ("self_name", "Soy estudiante", None, False, ""),
        ("self_name", "Soy feliz", None, False, ""),
        ("self_name", "Soy mal", None, False, ""),
        ("self_name", "Soy el mejor", None, False, ""),
        ("self_name", "Soy la mejor", None, False, ""),
        ("self_name", "Soy un crack", None, False, ""),

        # Sandy bloqueada
        ("self_name", "Me llamo Sandy", None, False, ""),
        ("self_name", "Soy sandi", None, False, ""),
        ("self_nick", "me llaman sandy", None, False, ""),

        # Negación cercana (None)
        ("self_name", "No me llamo Gabriel", None, False, ""),
        ("self_name", "Nunca me llamo Gabriel", None, False, ""),
        ("self_name", "No soy Gabriel", None, False, ""),
        ("self_name", "Jamás soy Gabriel", None, False, ""),
        ("self_name", "jamas me llamo Gabriel", None, False, ""),
        ("self_name", "no mi nombre es Gabriel", None, False, ""),
        ("self_name", "no no me llamo Gabriel", None, False, ""),

        # Negación + corrección
        ("self_name", "No me llamo Gabriel, me llamo Saul", "Saul", False, ""),
        ("self_name", "No soy Gabriel soy Alan", "Alan", False, ""),
        ("self_name", "No soy Gabriel, soy Alan", "Alan", False, ""),

        # Whisper errores
        ("self_name", "me yamo gabriel", None, False, ""),
        ("self_name", "mi nombree es gabriel", None, False, ""),
        ("self_name", "me llamo g4br13l", None, False, ""),

        # OJO: según tu policy actual, esto suele devolver "Gabriel" (corta el 2)
        ("self_name", "Me llamo Gabriel 2", "Gabriel", False, ""),

        # Frases largas (corte interno)
        ("self_name", "hola sandy mira estoy probando esto y bueno me llamo gabriel y necesito que lo registres", "gabriel", False, ""),
        ("self_name", "sandy hola yo soy gabriel y estoy en discord", None, False, ""),
        ("self_name", "sandy hola Soy Gabriel y estoy en discord", "Gabriel", False, ""),

        # =========================================================
        # MENTIONS
        # =========================================================
        ("mention", "Velok dijo que no venía", ["Velok"], False, ""),
        ("mention", "velok dijo que no venía", ["velok"], False, ""),
        ("mention", "Santiago cree que funciona", ["Santiago"], False, ""),
        ("mention", "Te presento a Santiago", ["Santiago"], False, ""),
        ("mention", "Conoce a Maria Jose", ["Maria Jose"], False, ""),
        ("mention", "Saluda a Juan Carlos", ["Juan Carlos"], False, ""),
        ("mention", "Este es Velok", ["Velok"], False, ""),
        ("mention", "Esta es Maria", ["Maria"], False, ""),
        ("mention", "Dile a Santiago que se apure", ["Santiago"], False, ""),
        ("mention", "Llama a Velok", ["Velok"], False, ""),
        ("mention", "Habla con Velok", ["Velok"], False, ""),

        # Este déjalo como XFAIL si aún no cortas bien "Según Gabriel eso..."
        ("mention", "Según Gabriel eso funciona", ["Gabriel"], True,
         "Si tu NAME_PATTERN aún traga tokens extra, aquí debería cortar a 'Gabriel'."),

        ("mention", "Velok dijo y luego Velok dijo otra cosa", ["Velok"], False, ""),

        ("mention", "Santiago mencionó que llega tarde", ["Santiago"], False, ""),
        ("mention", "Santiago menciono que llega tarde", ["Santiago"], False, ""),

        ("mention", "Este es el problema", [], False, ""),
        ("mention", "Te presento a mi amigo Sandy", [], False, ""),
        ("mention", "Te presento a el problema", [], False, ""),
        ("mention", "Conoce a un amigo Gabriel", [], False, ""),

        # XFAIL opcionales: depende de tu INVALID_HARD
        ("mention", "Habla con Velok mañana", ["Velok"], True, "Si no cortas 'mañana', se va a 'Velok mañana'."),
        ("mention", "Llama a Gabriel porfa", ["Gabriel"], True, "Si no cortas 'porfa', se va a 'Gabriel porfa'."),
        ("mention", "Te presento a mi amigo Santiago", ["Santiago"], True, "Si no soportas 'mi amigo' opcional, falla."),
        ("mention", "Dijo Velok que no venía", ["Velok"], True, "Si no soportas 'dijo {NAME}', falla."),
    ]

    print("\nIniciando pruebas de IntentDetector (v2)...\n")
    counts = {"pass": 0, "fail": 0, "xfail": 0, "upass": 0}

    for i, (method, text, expected, xfail, reason) in enumerate(tests, start=1):
        if method == "self_name":
            got = _detect_real_name(text)
        elif method == "self_nick":
            got = _detect_nickname(text)
        elif method == "mention":
            got = d.extract_mentioned_names(text)
        else:
            raise ValueError(f"Método desconocido: {method}")

        status = _assert_equal(i, method, text, expected, got, xfail, reason)
        counts[status] += 1

    total = len(tests)
    print("\nResumen:")
    print(f"  PASS : {counts['pass']}/{total}")
    print(f"  XFAIL: {counts['xfail']}/{total}")
    print(f"  UPASS: {counts['upass']}/{total}")
    print("\nListo.\n")


if __name__ == "__main__":
    run_tests()
