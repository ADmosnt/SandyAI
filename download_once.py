# download_once.py
# Run this once to download the TTS model for offline use.
# Usage: python download_once.py

import os
import torch

# Allowlist XTTS classes so torch.load works with weights_only=True (PyTorch 2.6+)
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
        print(f"⚠️ Allowlist XTTS: {e}")

_allowlist_xtts_globals()

from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"

print("⬇️ Descargando modelo TTS (xtts_v2)...")

try:
    TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("\n✅ Modelo TTS descargado correctamente.")
except Exception as e:
    print(f"\n❌ Error: {e}")
