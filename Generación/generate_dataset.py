import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pathlib import Path
from scipy.signal import welch, butter, filtfilt
from tqdm import tqdm

from model import UNet1D
from diffusion import GaussianDiffusion1D



# Configuración


WINDOW_SIZE    = 8192
TARGET_SR      = 4000
TIMESTEPS      = 1000

# Filtro pasabanda (igual que el preprocesamiento)
LOWCUT  = 25
HIGHCUT = 450
ORDER   = 4

# El orden define el índice de subtipo con el que se entrenó el modelo
CLASES = [
    "holosystolic_harsh",
    "holosystolic_blowing",
    "early_blowing",
    "early_harsh",
    "mid_harsh",
]
CLASE_A_IDX = {c: i for i, c in enumerate(CLASES)}

# Proporción media de cada estado (1=S1, 2=sístole, 3=S2, 4=diástole) por subtipo,
# medida en las máscaras de entrenamiento
PROPORCIONES_CLASE = {
    "holosystolic_harsh"  : {1: 0.224, 2: 0.212, 3: 0.181, 4: 0.383},
    "holosystolic_blowing": {1: 0.219, 2: 0.226, 3: 0.178, 4: 0.377},
    "early_blowing"       : {1: 0.245, 2: 0.235, 3: 0.192, 4: 0.328},
    "early_harsh"         : {1: 0.239, 2: 0.233, 3: 0.184, 4: 0.344},
    "mid_harsh"           : {1: 0.234, 2: 0.247, 3: 0.172, 4: 0.347},
}
VARIACION_PROP = 0.15   # ±15 % de variación aleatoria en cada proporción

N_POR_CLASE = 100
BPM_MIN     = 77.9     # P5 del conjunto de entrenamiento
BPM_MAX     = 136.3    # P95 del conjunto de entrenamiento
BATCH       = 25
SEED        = 42
NPERSEG     = 512

FILE_PATH      = Path(__file__).resolve()
PROJECT_ROOT   = FILE_PATH.parent.parent
CKPT_PATH      = PROJECT_ROOT / "checkpoints" / "ddpm_ablacion_inconv" / "ckpt_secuencial_mejor.pt"
PROCESADOS_DIR = PROJECT_ROOT / "data" / "positivos" / "procesados"
OUTPUT_DIR     = PROJECT_ROOT / "data" / "Dataset sintetico ablacion"



# Utilidades


def construir_mascara_bpm(clase: str, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """
    Máscara de estados de WINDOW_SIZE muestras (etiquetas 1-4) para un ritmo dado.
    Cada proporción del subtipo varía ±VARIACION_PROP y se renormaliza; el ciclo
    S1 → sístole → S2 → diástole se repite hasta llenar la ventana, iniciando en S1.
    """
    dur_ciclo_s = 60.0 / bpm
    props_base  = PROPORCIONES_CLASE[clase]
    orden       = [1, 2, 3, 4]

    props_var = {}
    for fase in orden:
        variacion = rng.uniform(-VARIACION_PROP, VARIACION_PROP)
        props_var[fase] = max(0.05, props_base[fase] * (1 + variacion))
    total_prop = sum(props_var.values())
    props_norm = {f: v / total_prop for f, v in props_var.items()}

    # Duración de cada estado en muestras; la diástole completa el ciclo
    dur_muestras, total_asignado = {}, 0
    for i, fase in enumerate(orden):
        if i < len(orden) - 1:
            dur_muestras[fase] = max(1, round(props_norm[fase] * dur_ciclo_s * TARGET_SR))
            total_asignado += dur_muestras[fase]
        else:
            dur_ciclo_muestras = round(dur_ciclo_s * TARGET_SR)
            dur_muestras[fase] = max(1, dur_ciclo_muestras - total_asignado)

    mask = np.zeros(WINDOW_SIZE, dtype=np.int64)
    idx = 0
    while idx < WINDOW_SIZE:
        for fase in orden:
            fin = min(idx + dur_muestras[fase], WINDOW_SIZE)
            mask[idx:fin] = fase
            idx = fin
            if idx >= WINDOW_SIZE:
                break
    return mask


def aplicar_filtro(audio: np.ndarray, sr: int) -> np.ndarray:
    """Butterworth pasabanda de orden 4 (25-450 Hz), fase cero con filtfilt."""
    if len(audio) < 3 * ORDER * 2:
        return audio
    nyquist = 0.5 * sr
    b, a = butter(ORDER, [LOWCUT / nyquist, HIGHCUT / nyquist], btype="band")
    return filtfilt(b, a, audio)


def mascara_a_tsv(mask: np.ndarray, sr: int = TARGET_SR) -> pd.DataFrame:
    """Agrupa muestras consecutivas con la misma etiqueta (igual que 6_segmentacion.py)."""
    filas, actual, inicio = [], mask[0], 0
    for i in range(1, len(mask)):
        if mask[i] != actual:
            filas.append({"start": round(inicio / sr, 6), "end": round(i / sr, 6),
                          "label": int(actual)})
            actual, inicio = mask[i], i
    filas.append({"start": round(inicio / sr, 6), "end": round(len(mask) / sr, 6),
                  "label": int(actual)})
    return pd.DataFrame(filas)


def log_psd(audio: np.ndarray) -> np.ndarray:
    """log10 de la PSD en 25-450 Hz normalizada por área."""
    f, p = welch(audio, fs=TARGET_SR, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    p = p[(f >= LOWCUT) & (f <= HIGHCUT)]
    return np.log10(p / p.sum() + 1e-12)


def ref_log_psd(clase: str) -> np.ndarray:
    """log10 de la PSD media de los segmentos de entrenamiento del subtipo."""
    wavs = sorted((PROCESADOS_DIR / "train" / clase / "secuencial").glob("*.wav"))
    psds = [10 ** log_psd(sf.read(w)[0].astype(np.float64)) for w in wavs]
    return np.log10(np.mean(psds, axis=0) + 1e-12)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])



# Generación


def main():
    existentes = list(OUTPUT_DIR.rglob("*.wav")) if OUTPUT_DIR.exists() else []
    if existentes:
        print(f" [ALTO] {OUTPUT_DIR} ya contiene {len(existentes)} audios.")
        print("        Bórrala antes de generar para no mezclar bancos.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    model = UNet1D().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    diffusion = GaussianDiffusion1D(timesteps=TIMESTEPS, device=device)

    print(f"\n Checkpoint : {CKPT_PATH.name} (época {ckpt['epoch']})")
    print(f" Subtipos   : {N_POR_CLASE} por subtipo, sin filtro de aceptación")
    print(f" BPM        : uniforme en [{BPM_MIN}, {BPM_MAX}]")
    print(f" Salida     : {OUTPUT_DIR}\n")

    meta = []
    for clase in CLASES:
        carpeta = OUTPUT_DIR / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        ref = ref_log_psd(clase)
        n = 0
        pbar = tqdm(total=N_POR_CLASE, desc=f" {clase:<22}")

        while n < N_POR_CLASE:
            b = min(BATCH, N_POR_CLASE - n)
            bpms = rng.uniform(BPM_MIN, BPM_MAX, size=b)
            masks = np.stack([construir_mascara_bpm(clase, bpm, rng) for bpm in bpms])
            mask_t = torch.tensor(masks, dtype=torch.long, device=device).unsqueeze(1)
            idx = torch.full((b,), CLASE_A_IDX[clase], dtype=torch.long, device=device)

            model.eval()   # diffusion.sample deja el modelo en modo train al terminar
            with torch.no_grad():
                x = diffusion.sample(model=model, mask_condition=mask_t, clase_idx=idx,
                                     n_samples=b, seq_length=WINDOW_SIZE)
            x = x.squeeze(1).cpu().numpy().astype(np.float64)

            for k in range(b):
                audio = aplicar_filtro(x[k], TARGET_SR)
                m = np.max(np.abs(audio))
                if m > 1e-8:
                    audio = audio / m
                nombre = f"sint_abl_{clase}_s{n:04d}_bpm{bpms[k]:.0f}"
                sf.write(carpeta / f"{nombre}.wav", audio.astype(np.float32), TARGET_SR)
                mascara_a_tsv(masks[k]).to_csv(carpeta / f"{nombre}.tsv", sep="\t",
                                               header=False, index=False)
                meta.append({"archivo": f"{clase}/{nombre}.wav", "clase": clase,
                             "bpm": round(float(bpms[k]), 1),
                             "corr_log_psd_train": round(pearson(log_psd(audio), ref), 4)})
                n += 1
            pbar.update(b)
        pbar.close()

    df = pd.DataFrame(meta)
    df.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

    print("\n Resumen (correlación espectral log10 frente a entrenamiento, solo informativa):")
    print(df.groupby("clase")["corr_log_psd_train"].describe()[["count", "mean", "min", "max"]]
          .round(3).to_string())
    print(f"\n Banco guardado en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
