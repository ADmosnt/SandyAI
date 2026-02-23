from speechbrain.inference import SpeakerRecognition
import os
import shutil

def download_model():
    print("⬇️ Descargando modelo de voz para uso OFFLINE...")
    
    # Creamos una carpeta local para el modelo
    local_model_dir = "sandy_voices_db/model_source"
    
    # Instanciamos para que descargue todo en la carpeta especificada
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=local_model_dir,
        run_opts={"device": "cpu"}
    )
    
    print(f"✅ Modelo descargado exitosamente en: {local_model_dir}")
    print("Ahora Sandy puede cargar esto sin internet y sin subprocesos.")

if __name__ == "__main__":
    download_model()