import numpy as np
import soundfile as sf
from pathlib import Path



# Configuración


PERCENTIL = 99.8   # umbral de clipping

CLASES = [
    "holosystolic_harsh",
    "holosystolic_blowing",
    "early_blowing",
    "early_harsh",
    "mid_harsh",
]



# Función principal


def aplicar_clipping_positivos():

    FILE_PATH    = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    INPUT_DIR  = PROJECT_ROOT / "data" / "positivos" / "raw_recortado"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "positivos" / "raw_clipping"

    

    archivos_wav = list(INPUT_DIR.rglob("*.wav"))

    if not archivos_wav:
        print(f" Error: No se encontraron archivos en {INPUT_DIR}")
        return

    print(f" Archivos encontrados: {len(archivos_wav)}\n")

    procesados  = 0
    descartados = 0
    conteo_por_clase = {c: 0 for c in CLASES}

    for wav_path in archivos_wav:

        # Determinar clase desde la ruta
        clase = None
        for c in CLASES:
            if c in wav_path.parts:
                clase = c
                break

        # Replicar estructura de carpetas en output
        ruta_relativa  = wav_path.relative_to(INPUT_DIR)
        carpeta_salida = OUTPUT_DIR / ruta_relativa.parent
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        # Leer audio
        audio, sr = sf.read(wav_path)
        audio     = audio.astype(np.float64)

        # Eliminar offset DC
        audio = audio - np.mean(audio)

        # Calcular umbral
        p = np.percentile(np.abs(audio), PERCENTIL)

        if p == 0:
            print(f"  [WARN] Senal plana, descartada: {wav_path.name}")
            descartados += 1
            continue

        # Aplicar clipping y normalizar a ±1
        audio = np.clip(audio, -p, p)
        audio = audio / p

        # Guardar audio
        sf.write(carpeta_salida / wav_path.name, audio, sr)

        # Copiar .tsv sin modificarlo
        tsv_path = wav_path.with_suffix(".tsv")
        if tsv_path.exists():
            out_tsv = carpeta_salida / tsv_path.name
            out_tsv.write_bytes(tsv_path.read_bytes())

        procesados += 1
        if clase:
            conteo_por_clase[clase] += 1

    # Resumen
    print("Clipping completado\n")
    print(f"   Archivos procesados  : {procesados}")
    print(f"   Archivos descartados : {descartados}")
    print(f"\n Archivos por clase:")
    for clase, n in sorted(conteo_por_clase.items()):
        print(f"   {clase:<25} : {n}")
    


if __name__ == "__main__":
    aplicar_clipping_positivos()