import os
import torch

# --- INICIO DEL PARCHE PIRATA PARA PYTORCH 2.6+ ---
# Guardamos la función original de carga
_original_load = torch.load

# Definimos una función que fuerza weights_only=False
def _hack_torch_load(*args, **kwargs):
    # Si la librería intenta pasar weights_only, lo ignoramos o lo forzamos a False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

# Reemplazamos la función de PyTorch con la nuestra
torch.load = _hack_torch_load
print("🔧 Parche de 'weights_only' aplicado con éxito.")
# --- FIN DEL PARCHE ---

from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"

print("⬇️ VERIFICANDO MODELO...")

try:
    # Esto ahora usará nuestra función de carga hackeada
    TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("\n✅ ¡LISTO! Modelo cargado sin errores de seguridad.")
except Exception as e:
    print(f"\n❌ Error: {e}")