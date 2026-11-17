import pandas as pd
import soundfile as sf
from pathlib import Path



# Configuración

MIN_FILAS_CHUNK = 4


CLASS_MAP = {
    ("holosystolic",   "harsh")  : "holosystolic_harsh",
    ("holosystolic",   "blowing"): "holosystolic_blowing",
    ("early-systolic", "blowing"): "early_blowing",
    ("early-systolic", "harsh")  : "early_harsh",
    ("mid-systolic",   "harsh")  : "mid_harsh",
}



# Utilidades


def _get_clase(row) -> str | None:
    """Determina la subtipo de un paciente desde una fila del Excel."""
    timing  = str(row.get("Systolic murmur timing",  "")).strip().lower()
    quality = str(row.get("Systolic murmur quality", "")).strip().lower()
    return CLASS_MAP.get((timing, quality), None)


def _parse_murmur_locations(raw_value) -> set:
    """'TV+MV' - {'TV', 'MV'} | nan - set()"""
    if pd.isna(raw_value):
        return set()
    return {loc.strip().upper() for loc in str(raw_value).split("+")}


def _extract_location(filename: str) -> str:
    """'12345_TV.wav' - 'TV'"""
    parts = Path(filename).stem.split("_")
    return parts[-1].upper() if len(parts) >= 2 else ""


def _procesar_wav(wav_path: Path, audio, sr: int, df_tsv: pd.DataFrame,
                  carpeta_salida: Path) -> tuple:
    """
    sin label 0 anclados en S1.

      - Un chunk comienza SOLO cuando aparece label=1 (S1)
      - Un chunk se cierra cuando aparece label=0 (ruido)
      - Labels 2/3/4 antes del primer S1 se descartan
      - Si hay ruido en medio, se generan múltiples chunks

    Retorna (chunks_validos, chunks_cortos).
    """
    chunks        = []
    current_chunk = []

    for _, row in df_tsv.iterrows():
        etiqueta = int(row["label"])
        if etiqueta == 0:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
        else:
            if not current_chunk:
                if etiqueta == 1:
                    current_chunk.append(row)
            else:
                current_chunk.append(row)

    if current_chunk:
        chunks.append(current_chunk)

    if not chunks:
        return 0, 0

    chunks_validos = 0
    chunks_cortos  = 0

    for idx, chunk in enumerate(chunks):
        if len(chunk) < MIN_FILAS_CHUNK:
            chunks_cortos += 1
            continue

        df_chunk = pd.DataFrame(chunk).copy()

        tiempo_inicio = df_chunk.iloc[0]["start"]
        tiempo_fin    = df_chunk.iloc[-1]["end"]

        # Recalibrar tiempos desde cero
        df_chunk["start"] = (df_chunk["start"] - tiempo_inicio).round(6)
        df_chunk["end"]   = (df_chunk["end"]   - tiempo_inicio).round(6)
        df_chunk["label"] = df_chunk["label"].astype(int)

        muestra_inicio = round(tiempo_inicio * sr)
        muestra_fin    = min(round(tiempo_fin * sr), len(audio))
        audio_recortado = audio[muestra_inicio:muestra_fin]

        # Nombre: conservar original si hay un solo chunk
        if len(chunks) == 1:
            base_name = wav_path.stem
        else:
            base_name = f"{wav_path.stem}_parte_{idx + 1}"

        sf.write(carpeta_salida / f"{base_name}.wav", audio_recortado, sr)
        df_chunk.to_csv(carpeta_salida / f"{base_name}.tsv",
                        sep="\t", header=False, index=False)

        chunks_validos += 1

    return chunks_validos, chunks_cortos



# Función principal


def recortar_ruido_positivos():

    FILE_PATH    = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    RAW_DIR    = PROJECT_ROOT / "data" / "raw"
    EXCEL_PATH = RAW_DIR / "training_data_filtrada.xlsx"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "positivos" / "raw_recortado"

    


    # 1. Leer Excel y construir índice de pacientes


    if not EXCEL_PATH.exists():
        print(f" Error: No se encontro {EXCEL_PATH}")
        return

    df_excel = pd.read_excel(EXCEL_PATH)

    # Construir índice
    patient_index = {}
    descartados_excel = 0

    for _, row in df_excel.iterrows():
        pid   = str(row["Patient ID"])
        clase = _get_clase(row)
        if clase is None:
            descartados_excel += 1
            continue
        patient_index[pid] = {
            "clase"            : clase,
            "murmur_locations" : _parse_murmur_locations(row["Murmur locations"]),
        }

    print(f" Pacientes en Excel          : {len(df_excel)}")
    print(f" Pacientes clasificados      : {len(patient_index)}")
    if descartados_excel:
        print(f" Pacientes fuera de CLASS_MAP: {descartados_excel}")


    # 2. Procesar archivos


    # Contadores
    archivos_procesados   = 0
    partes_generadas      = 0
    chunks_cortos_total   = 0
    archivos_sin_tsv      = 0
    archivos_sin_chunks   = 0
    archivos_sin_soplo    = 0   # localización no está en Murmur locations
    pacientes_no_en_raw   = 0

    conteo_por_clase = {c: 0 for c in CLASS_MAP.values()}

    for pid, info in patient_index.items():
        clase            = info["clase"]
        murmur_locations = info["murmur_locations"]

        # Buscar archivos del paciente en raw/
        archivos_paciente = list(RAW_DIR.rglob(f"{pid}_*.wav"))

        if not archivos_paciente:
            pacientes_no_en_raw += 1
            continue

        for wav_path in archivos_paciente:
            tsv_path = wav_path.with_suffix(".tsv")

            if not tsv_path.exists():
                archivos_sin_tsv += 1
                continue

            # Verificar que localización tiene soplo confirmado
            location = _extract_location(wav_path.name)
            if location not in murmur_locations:
                archivos_sin_soplo += 1
                continue

            # Carpeta de salida
            carpeta_salida = OUTPUT_DIR / clase
            carpeta_salida.mkdir(parents=True, exist_ok=True)

            # Leer audio y TSV
            try:
                audio, sr = sf.read(wav_path)
                df_tsv = pd.read_csv(tsv_path, sep="\t", header=None,
                                     names=["start", "end", "label"])
            except Exception as e:
                print(f"  [ERROR] Leyendo {wav_path.name}: {e}")
                continue

            # Procesar chunks
            validos, cortos = _procesar_wav(
                wav_path, audio, sr, df_tsv, carpeta_salida
            )

            if validos == 0 and cortos == 0:
                archivos_sin_chunks += 1
                continue

            chunks_cortos_total += cortos
            partes_generadas    += validos
            conteo_por_clase[clase] += validos

            if validos > 0:
                archivos_procesados += 1


    # 3. Resumen


    print("\n Resumen:")
    print(f"   Archivos procesados        : {archivos_procesados}")
    print(f"   Fragmentos limpios creados : {partes_generadas}")
    print(f"   Chunks cortos descartados  : {chunks_cortos_total}")
    print(f"   Archivos sin .tsv          : {archivos_sin_tsv}")
    print(f"   Archivos sin S1 valido     : {archivos_sin_chunks}")
    print(f"   Archivos sin soplo en loc. : {archivos_sin_soplo}")
    print(f"   Pacientes no en raw/       : {pacientes_no_en_raw}")

    print(f"\n Fragmentos por clase:")
    for clase, n in sorted(conteo_por_clase.items()):
        print(f"   {clase:<25} : {n}")

    


if __name__ == "__main__":
    recortar_ruido_positivos()