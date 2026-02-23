import pyaudio
import sounddevice as sd

print("\n--- 🎧 DISPOSITIVOS DE ENTRADA (Micrófonos) - PyAudio ---")
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        name = p.get_device_info_by_host_api_device_index(0, i).get('name')
        print(f"ID {i}: {name}")

print("\n--- 🔊 DISPOSITIVOS DE SALIDA (Altavoces) - SoundDevice ---")
print(sd.query_devices())
print("\nBusca los que dicen 'NVIDIA Broadcast' o tus audífonos.")