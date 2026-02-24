# app/core/personality.py

import json
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERSONALITY_FILE = os.path.join(_BASE_DIR, "domain", "personality.json")

_cache = None


def load_personality() -> dict | None:
    """Return the personality dict (cached after first load)."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(_PERSONALITY_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
        return _cache
    except Exception as e:
        print(f"Error al cargar la personalidad: {e}")
        return None


def build_system_prompt() -> str:
    """Build a system prompt string from personality.json."""
    p = load_personality()
    if not p:
        return "Eres Sandy, una asistente de IA avanzada."

    info = p.get("basic_info", {})
    traits = p.get("personality_traits", {})
    rules = p.get("response_rules", {})
    patterns = p.get("response_patterns", {})

    name = info.get("name", "Sandy")
    nationality = info.get("nationality", "")
    age = info.get("age", "")

    core = traits.get("core_traits", [])
    temperament = traits.get("temperament", "")
    alignment = traits.get("moral_alignment", "")

    memory_rules = rules.get("memory_behavior", [])
    output_rules = rules.get("output_format", [])

    modismos = patterns.get("modismos", [])

    prompt = f"Eres {name}, una asistente de IA avanzada"
    if nationality:
        prompt += f" {nationality.lower()}"
    if age:
        prompt += f" de {age} años"
    prompt += ".\n\n"

    if core:
        prompt += "PERSONALIDAD:\n"
        for t in core:
            prompt += f"- {t}\n"
        if temperament:
            prompt += f"- Temperamento: {temperament}\n"
        if alignment:
            prompt += f"- Alineamiento moral: {alignment}\n"
        prompt += "\n"

    if memory_rules or output_rules:
        prompt += "REGLAS:\n"
        for r in memory_rules:
            prompt += f"- {r}\n"
        for r in output_rules:
            prompt += f"- {r}\n"
        prompt += "\n"

    if modismos:
        prompt += f"MODISMOS QUE USAS: {', '.join(modismos[:20])}\n"

    return prompt.strip()
