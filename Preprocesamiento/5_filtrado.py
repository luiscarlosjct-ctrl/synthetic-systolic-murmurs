import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt



# Configuración


LOWCUT  = 25    
HIGHCUT = 450   
ORDER   = 4     

CLASES = [
    "holosystolic_harsh",
    "holosystolic_blowing",
    "early_blowing",
    "early_harsh",
    "mid_harsh",
]



# Función principal


def aplicar_filtro_positivos():

    FILE_PATH    = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    INPUT_DIR  = PROJECT_ROOT / "data" / "positivos" / "raw_clipping"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "positivos" / "raw_filtrado"

    archivos_wav = list(INPUT_DIR.rglob("*.wav"))

    if not archivos_wav:
        print(f" Error: No se encontraron archivos en {INPUT_DIR}")
        return



    procesados       = 0
    descartados      = 0
    conteo_por_clase = {c: 0 for c in CLASES}

    for wav_path in archivos_wav:

        # Determinar clase desde la ruta
        clase = None
        for c in CLASES:
            if c in wav_path.parts:
                clase = c
                break

        # Replicar estructura de carpetas en destino
        ruta_relativa  = wav_path.relative_to(INPUT_DIR)
        carpeta_salida = OUTPUT_DIR / ruta_relativa.parent
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        # Leer audio
        audio, sr = sf.read(wav_path)
        audio     = audio.astype(np.float64)

        # Validar longitud mínima para filtfilt
        min_len = 3 * ORDER * 2
        if len(audio) < min_len:
            print(f"  [WARN] Demasiado corto, descartado: {wav_path.name} "
                  f"({len(audio)} muestras < {min_len})")
            descartados += 1
            continue

        # Diseñar y aplicar filtro
        nyquist = 0.5 * sr
        b, a    = butter(ORDER, [LOWCUT / nyquist, HIGHCUT / nyquist],
                         btype="band")
        audio_filtrado = filtfilt(b, a, audio)

        # Guardar audio filtrado
        sf.write(carpeta_salida / wav_path.name, audio_filtrado, sr)

        # Copiar .tsv sin modificarlo
        tsv_path = wav_path.with_suffix(".tsv")
        if tsv_path.exists():
            out_tsv = carpeta_salida / tsv_path.name
            out_tsv.write_bytes(tsv_path.read_bytes())

        procesados += 1
        if clase:
            conteo_por_clase[clase] += 1

    # Resumen
    print("Filtrado completado\n")
    print(f"   Archivos procesados  : {procesados}")
    print(f"   Archivos descartados : {descartados}")
    print(f"\n Archivos por clase:")
    for clase, n in sorted(conteo_por_clase.items()):
        print(f"   {clase:<25} : {n}")
    


if __name__ == "__main__":
    aplicar_filtro_positivos()