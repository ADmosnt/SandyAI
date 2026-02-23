# app/services/brain/intent_logger.py

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path("logs/intent_detection")

def _write_event(event: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"events_{datetime.now().strftime('%Y%m')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def log_miss(text: str, speaker_id: Optional[str], hints: Optional[Dict[str, Any]] = None) -> None:
    event = {
        "ts": datetime.now().isoformat(),
        "type": "miss",
        "speaker_id": speaker_id,
        "text": text,
        "hints": hints or {},
    }
    _write_event(event)

def log_fallback(text: str, speaker_id: Optional[str], detector: Any, llm: Any, decision: str) -> None:
    event = {
        "ts": datetime.now().isoformat(),
        "type": "fallback_used",
        "speaker_id": speaker_id,
        "text": text,
        "detector": str(detector),
        "llm": str(llm),
        "decision": decision,
    }
    _write_event(event)

def log_ambiguity(text: str, speaker_id: Optional[str], candidates: list[str]) -> None:
    event = {
        "ts": datetime.now().isoformat(),
        "type": "ambiguity",
        "speaker_id": speaker_id,
        "text": text,
        "candidates": candidates,
    }
    _write_event(event)
