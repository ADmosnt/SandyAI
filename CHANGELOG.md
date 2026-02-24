# CHANGELOG - SandyAI

## [1.0.0] - 2026-02-24 - Limpieza y consolidacion del proyecto

### Contexto: Analisis de codigo legado

Se realizo un analisis completo del repositorio para separar el codigo activo
(main_local.py y sus dependencias) del codigo muerto heredado del proyecto
original "Sandy" (bot de Twitch/FastAPI).

**Hallazgos principales:**

- El proyecto contenia dos sistemas completamente separados:
  1. **main_local.py** (activo): Asistente de voz local con Gemini, speaker ID,
     barge-in, perfiles multi-usuario y memoria episodica.
  2. **Sandy original** (muerto): Infraestructura Twitch/FastAPI con WebSockets,
     moderacion de chat y VTuber prompts. No se ejecutaba ni estaba conectada.

- 2 archivos con errores de import que impedian su carga:
  - `gemini_router.py` importaba `GeminiServicesUseCase` (no existia)
  - `gemini_services.py` importaba `response_sandy_shandrew` (no existia)

- `requirements_backup.txt` estaba corrupto (codificacion UTF-16 con espacios
  entre caracteres) y tenia versiones completamente diferentes al entorno real.

### Archivos eliminados (codigo muerto Twitch/FastAPI)

```
app/main.py                              - Stub FastAPI vacio
app/core/config.py                       - Credenciales Twitch
app/core/bannedWords.py                  - Carga de banned_words.json
app/core/ports/websocket_port.py         - Interfaz abstracta WebSocket
app/domain/messages.py                   - Constante de mensaje de bienvenida
app/domain/exceptions.py                 - EventSubError exception
app/domain/banned_words.json             - Lista de palabras prohibidas
app/models/ProfileModel.py               - Schema Pydantic de broadcaster
app/models/twitch_auth_model.py          - Modelos OAuth Twitch
app/models/tokens_model.py               - Schema de tokens
app/models/message_model.py              - Schema de mensajes FastAPI
app/models/websocket_models.py           - Tipos de mensajes WebSocket
app/services/moderator.py                - Moderacion de chat
app/services/application/status_service.py - Endpoint de status
app/controllers/http/gemini_router.py    - Router POST /gemini (ROTO)
app/controllers/http/test_router.py      - Router GET /
app/controllers/websocket/websocket_server.py - Manager WebSocket
app/adapters/gemini_services.py          - Adaptador Gemini (ROTO)
app/adapters/websocket_adapter.py        - Adaptador WebSocket
app/config/cors.py                       - Configuracion CORS
requirements_backup.txt                  - Backup corrupto de dependencias
```

Directorios eliminados (quedaron vacios):
`app/models/`, `app/controllers/`, `app/adapters/`, `app/config/`,
`app/core/ports/`, `app/services/application/`

### Archivos conservados del proyecto original

- `app/domain/prompts.py` - Prompts de VTuber/moderacion (referencia futura)
- `app/domain/personality.json` - Personalidad completa de Sandy
- `app/core/personality.py` - Cargador de personalidad

### Cambios en personality.json

Se agrego la seccion `response_rules` con las reglas de comportamiento de
memoria y formato de salida que antes estaban hardcodeadas en gemini.py:

```json
"response_rules": {
    "memory_behavior": [
        "NO menciones explicitamente 'segun mi base de datos' o 'recuerdo que...'",
        "Actua como si naturalmente lo recordaras",
        "Usa el perfil del usuario para personalizar respuestas"
    ],
    "output_format": [
        "Respuestas concisas y conversacionales",
        "Solo texto plano limpio...",
        "No uses emojis, simbolos...",
        "No respondas usando palabras japonesas"
    ]
}
```

### Cambios en app/core/personality.py

Reescrito para:
- Parsear el JSON (antes solo leia texto crudo)
- Generar el system prompt dinamicamente desde personality.json
- Incluir personalidad, reglas de memoria, formato de salida y modismos
- Cachear el JSON despues de la primera carga

### Cambios en app/services/gemini.py

- El `SYSTEM_PROMPT` hardcodeado fue reemplazado por `build_system_prompt()`
  que genera el prompt desde personality.json
- Se importa `from app.core.personality import build_system_prompt`

### Cambios en requirements.txt

Paquetes eliminados (solo eran necesarios para Twitch/FastAPI/WebSocket):
`twitchAPI`, `pyvts`, `bcrypt`, `oauthlib`, `requests-oauthlib`,
`pvporcupine`, `openwakeword`, `RealtimeSTT`, `realtimetts`,
`python-engineio`, `python-socketio`, `simple-websocket`, `bidict`,
`uvicorn`, `httptools`, `watchfiles`, `wsproto`, `mss`, `opencv-python`,
`sseclient-py`, `git-filter-repo`, `kubernetes`, `durationpy`

De 218 paquetes a 194.

### Estructura final del proyecto

```
SandyAI/
├── main_local.py                  [ENTRY POINT]
├── requirements.txt               [DEPENDENCIAS]
├── CHANGELOG.md                   [ESTE ARCHIVO]
├── .pre-commit-config.yaml        [HOOKS DE CALIDAD]
├── download_once.py               [UTIL: patch PyTorch + verificacion TTS]
├── get_devices.py                 [UTIL: enumeracion de dispositivos audio]
├── test.py                        [UTIL: descarga modelo speaker recognition]
├── test_intent.py                 [UTIL: tests de intent detector]
├── sandy_ref.wav                  [REQUERIDO: voz de referencia para TTS]
├── app/
│   ├── core/
│   │   └── personality.py         [CARGADOR + BUILDER de system prompt]
│   ├── domain/
│   │   ├── personality.json       [PERSONALIDAD COMPLETA DE SANDY]
│   │   └── prompts.py             [PROMPTS DE REFERENCIA (VTuber/mod)]
│   └── services/
│       ├── gemini.py              [CLIENTE GEMINI + STREAMING]
│       ├── audio/
│       │   ├── listening_service.py   [CAPTURA + VAD + WHISPER]
│       │   ├── tts_service.py         [SINTESIS XTTS + BARGE-IN]
│       │   └── speaker_service.py     [EMBEDDINGS + IDENTIFICACION]
│       └── brain/
│           ├── intent_detector.py     [DETECCION DE INTENCIONES]
│           ├── intent_logger.py       [LOG DE EVENTOS]
│           ├── memory_service.py      [MEMORIA EPISODICA - CHROMADB]
│           ├── profile_service.py     [PERFILES MULTI-USUARIO]
│           └── conversation_context.py [CONTEXTO DE CONVERSACION]
```
