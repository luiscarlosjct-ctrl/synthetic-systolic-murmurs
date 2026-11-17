import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm



# Configuración


WINDOW_SIZE  = 8192     # muestras — 2 ciclos cardíacos a 4000 Hz (~2.048 s)
TARGET_SR    = 4000     # Hz

CLASES = [
    "holosystolic_harsh",
    "holosystolic_blowing",
    "early_blowing",
    "early_harsh",
    "mid_harsh",
]



# Utilidades


def get_s1_starts(mask: np.ndarray) -> list:
    """Detecta el índice de inicio de cada S1 en la máscara."""
    starts, in_s1 = [], False
    for i, val in enumerate(mask):
        if val == 1 and not in_s1:
            starts.append(i)
            in_s1 = True
        elif val != 1:
            in_s1 = False
    return starts


def construir_mascara(df_tsv: pd.DataFrame, audio_len: int) -> np.ndarray:
    """Construye la máscara de etiquetas sobre toda la señal."""
    mask = np.zeros(audio_len, dtype=int)
    for _, row in df_tsv.iterrows():
        s = max(0, int(row["start"] * TARGET_SR))
        e = min(audio_len, int(row["end"] * TARGET_SR))
        if e > s:
            mask[s:e] = int(row["label"])
    return mask


def validar_segmento(seg_mask: np.ndarray) -> tuple:
    """
    Valida que el segmento contenga los 4 estados cardíacos completos.

    Criterios:
      - S1 (1), Sístole (2), S2 (3), Diáatole: presencia obligatoria
    

    Retorna (valido: bool, motivo: str).
    """
    fases = set(np.unique(seg_mask))
    for fase in [1, 2, 3, 4]:
        if fase not in fases:
            return False, f"falta fase {fase}"
    return True, "ok"


def generar_tsv_segmento(seg_mask: np.ndarray, sr: int) -> pd.DataFrame:
    """
    Genera el TSV recalibrado desde cero para un segmento.
    Agrupa muestras consecutivas con el mismo label en filas continuas.
    """
    filas = []
    label_actual = seg_mask[0]
    inicio       = 0

    for i in range(1, len(seg_mask)):
        if seg_mask[i] != label_actual:
            filas.append({
                "start": round(inicio / sr, 6),
                "end"  : round(i     / sr, 6),
                "label": int(label_actual),
            })
            label_actual = seg_mask[i]
            inicio       = i

    # Último bloque
    filas.append({
        "start": round(inicio          / sr, 6),
        "end"  : round(len(seg_mask)   / sr, 6),
        "label": int(label_actual),
    })

    return pd.DataFrame(filas)


def extraer_patient_location(filename: str) -> tuple:
    """
    '49823_PV.wav' → ('49823', 'PV')
    '49823_PV_parte_1.wav' → ('49823', 'PV')
    """
    stem  = Path(filename).stem
    parts = stem.split("_")
    pid   = parts[0]
    loc   = parts[1] if len(parts) >= 2 else "XX"
    return pid, loc



# Función principal


def segmentar_positivos():

    FILE_PATH    = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    INPUT_DIR  = PROJECT_ROOT / "data" / "positivos" / "raw_filtrado"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "positivos" / "segmentados"

    print(f" Pipeline  : POSITIVOS")
    print(f" Entrada   : {INPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print(f" Salida    : {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print(f" Ventana   : {WINDOW_SIZE} muestras ({WINDOW_SIZE/TARGET_SR:.3f} s)")
    print(f" Modos     : secuencial + solapado\n")

    # Contadores globales
    resumen = {
        clase: {
            "secuencial": {"aceptados": 0, "rechazados_fase": 0, "rechazados_rango": 0},
            "solapado"  : {"aceptados": 0, "rechazados_fase": 0, "rechazados_rango": 0},
        }
        for clase in CLASES
    }

    for clase in CLASES:
        clase_dir = INPUT_DIR / clase
        if not clase_dir.exists():
            print(f" [WARN] Carpeta no encontrada: {clase_dir.relative_to(PROJECT_ROOT)}")
            continue

        wav_files = list(clase_dir.glob("*.wav"))
        if not wav_files:
            print(f" [WARN] Sin archivos en {clase}")
            continue

        # Crear carpetas de salida
        dir_sec = OUTPUT_DIR / clase / "secuencial"
        dir_sol = OUTPUT_DIR / clase / "solapado"
        dir_sec.mkdir(parents=True, exist_ok=True)
        dir_sol.mkdir(parents=True, exist_ok=True)

        # Índice de segmentos por archivo para nombrado único
        idx_sec = {}
        idx_sol = {}

        for wav_path in tqdm(wav_files,
                             desc=f"[positivos|{clase}]",
                             leave=False):
            tsv_path = wav_path.with_suffix(".tsv")
            if not tsv_path.exists():
                continue

            pid, loc = extraer_patient_location(wav_path.name)
            key      = f"{pid}_{loc}"

            try:
                audio, sr = librosa.load(wav_path, sr=TARGET_SR)
                df_tsv    = pd.read_csv(tsv_path, sep="\t", header=None,
                                        names=["start", "end", "label"])
            except Exception as e:
                print(f"\n [ERROR] {wav_path.name}: {e}")
                continue

            mask      = construir_mascara(df_tsv, len(audio))
            s1_starts = get_s1_starts(mask)

            # MODO SECUENCIAL
            ultimo_fin = 0
            if key not in idx_sec:
                idx_sec[key] = 0

            for start_idx in s1_starts:
                if start_idx < ultimo_fin:
                    continue

                end_idx = start_idx + WINDOW_SIZE
                if end_idx > len(audio):
                    resumen[clase]["secuencial"]["rechazados_rango"] += 1
                    continue

                seg      = audio[start_idx:end_idx]
                seg_mask = mask[start_idx:end_idx]
                ultimo_fin = end_idx

                valido, motivo = validar_segmento(seg_mask)
                if not valido:
                    resumen[clase]["secuencial"]["rechazados_fase"] += 1
                    continue

                base_name = f"{pid}_{loc}_s{idx_sec[key]:03d}"
                sf.write(dir_sec / f"{base_name}.wav", seg, TARGET_SR)
                df_seg = generar_tsv_segmento(seg_mask, TARGET_SR)
                df_seg.to_csv(dir_sec / f"{base_name}.tsv",
                              sep="\t", header=False, index=False)

                resumen[clase]["secuencial"]["aceptados"] += 1
                idx_sec[key] += 1

            # MODO SOLAPADO
            if key not in idx_sol:
                idx_sol[key] = 0

            for start_idx in s1_starts:
                end_idx = start_idx + WINDOW_SIZE
                if end_idx > len(audio):
                    resumen[clase]["solapado"]["rechazados_rango"] += 1
                    continue

                seg      = audio[start_idx:end_idx]
                seg_mask = mask[start_idx:end_idx]

                valido, motivo = validar_segmento(seg_mask)
                if not valido:
                    resumen[clase]["solapado"]["rechazados_fase"] += 1
                    continue

                base_name = f"{pid}_{loc}_s{idx_sol[key]:03d}"
                sf.write(dir_sol / f"{base_name}.wav", seg, TARGET_SR)
                df_seg = generar_tsv_segmento(seg_mask, TARGET_SR)
                df_seg.to_csv(dir_sol / f"{base_name}.tsv",
                              sep="\t", header=False, index=False)

                resumen[clase]["solapado"]["aceptados"] += 1
                idx_sol[key] += 1


    # Resumen


    print("\n Resumen de segmentacion POSITIVOS:\n")
    print(f"   {'Clase':<25} {'Sec-OK':>7} {'Sec-Fase':>9} {'Sec-Rng':>8} "
          f"{'Sol-OK':>7} {'Sol-Fase':>9} {'Sol-Rng':>8}")
    print("   " + "-" * 77)

    total_sec = total_sol = 0
    for clase in CLASES:
        s = resumen[clase]["secuencial"]
        o = resumen[clase]["solapado"]
        total_sec += s["aceptados"]
        total_sol += o["aceptados"]
        print(f"   {clase:<25} {s['aceptados']:>7} {s['rechazados_fase']:>9} "
              f"{s['rechazados_rango']:>8} "
              f"{o['aceptados']:>7} {o['rechazados_fase']:>9} "
              f"{o['rechazados_rango']:>8}")

    print("   " + "-" * 77)
    print(f"   {'TOTAL':<25} {total_sec:>7} {' ':>9} {' ':>8} "
          f"{total_sol:>7}")
    


if __name__ == "__main__":
    segmentar_positivos()