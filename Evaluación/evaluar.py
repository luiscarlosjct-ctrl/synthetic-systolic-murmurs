from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import hilbert, welch

try:
    import librosa
    HAY_LIBROSA = True
except ImportError:
    HAY_LIBROSA = False



# RUTAS

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
PROCESADOS_DIR = PROJECT_ROOT / "data" / "positivos" / "procesados"   
HOLDOUT_PROC   = PROJECT_ROOT / "data" / "holdout" / "procesados"     
SINT_DIR       = PROJECT_ROOT / "data" / "Dataset sintetico ablacion"       
OUT_DIR        = PROJECT_ROOT / "outputs" / "evaluacion_ablacion"



# Configuración

CLASES = ["holosystolic_harsh", "holosystolic_blowing",
          "early_blowing", "early_harsh", "mid_harsh"]
MODO        = "secuencial"
TARGET_SR   = 4000
LOWCUT, HIGHCUT = 25, 450
NPERSEG_FEAT    = 512                      # PSD para correlación, MMD y centroides
NPERSEG_BANDAS  = 1024                     # PSD para energía por banda
BANDAS      = [(25, 100), (100, 200), (200, 300), (300, 450)]
ETIQ_BANDAS = ["25–100", "100–200", "200–300", "300–450"]

N_PARES_DTW = 1000
N_PERM_MMD  = 500
SEED        = 0



# Utilidades de señal

def leer_wav(path: Path) -> np.ndarray:
    audio, _ = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float64)


def mascara_desde_tsv(tsv: Path, n: int) -> np.ndarray:
    """Máscara de estados (1 S1, 2 sístole, 3 S2, 4 diástole)."""
    mask = np.zeros(n, dtype=np.int64)
    if not tsv.exists():
        return mask
    df = pd.read_csv(tsv, sep="\t", header=None, names=["start", "end", "label"])
    for _, r in df.iterrows():
        s, e = max(0, int(r["start"] * TARGET_SR)), min(n, int(r["end"] * TARGET_SR))
        if e > s:
            mask[s:e] = int(r["label"])
    return mask


def psd_banda(audio: np.ndarray) -> np.ndarray:
    """PSD en 25-450 Hz normalizada por área."""
    f, p = welch(audio, fs=TARGET_SR, nperseg=NPERSEG_FEAT, noverlap=NPERSEG_FEAT // 2)
    p = p[(f >= LOWCUT) & (f <= HIGHCUT)]
    return p / p.sum() if p.sum() > 0 else p


def log_psd(audio: np.ndarray) -> np.ndarray:
    return np.log10(psd_banda(audio) + 1e-12)


def pct_bandas(audio: np.ndarray) -> np.ndarray:
    f, p = welch(audio, fs=TARGET_SR, nperseg=NPERSEG_BANDAS, noverlap=NPERSEG_BANDAS // 2)
    e = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in BANDAS])
    return e / e.sum() * 100 if e.sum() > 0 else np.full(len(BANDAS), np.nan)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metricas_fase(audio: np.ndarray, mask: np.ndarray) -> dict:
    """
    Energía por fase del ciclo:
      contraste = potencia (S1+S2) / potencia (sístole+diástole)  → respeta la máscara
      sis_dia   = potencia sístole / potencia diástole              → intensidad del soplo
      tercio1-3 = reparto de la energía sistólica en tercios         → ubicación del soplo
    """
    p = audio ** 2
    pot = {f: p[mask == f].mean() if (mask == f).any() else np.nan for f in (1, 2, 3, 4)}
    tramos = np.flatnonzero(np.diff(np.r_[0, (mask == 2).astype(int), 0])).reshape(-1, 2)
    terc = np.zeros(3)
    for a, b in tramos:
        if b - a < 3:
            continue
        c = np.linspace(a, b, 4).astype(int)
        terc += [p[c[k]:c[k + 1]].mean() for k in range(3)]
    terc = terc / terc.sum() if terc.sum() > 0 else np.full(3, np.nan)
    return {"contraste": (pot[1] + pot[3]) / (pot[2] + pot[4]),
            "sis_dia": pot[2] / pot[4],
            "tercio1": terc[0], "tercio2": terc[1], "tercio3": terc[2]}



# Carga

def cargar(carpeta: Path) -> list:
    """Lista de dicts {audio, mask, pid} para los .wav de la carpeta."""
    out = []
    for w in sorted(carpeta.glob("*.wav")):
        audio = leer_wav(w)
        out.append({"audio": audio, "pid": w.stem.split("_")[0],
                    "mask": mascara_desde_tsv(w.with_suffix(".tsv"), len(audio))})
    return out


def ref_log(items: list) -> np.ndarray:
    """log10 de la media de las PSD normalizadas."""
    return np.log10(np.mean([psd_banda(it["audio"]) for it in items], axis=0) + 1e-12)



# DTW de envolventes

def envolvente(audio: np.ndarray) -> np.ndarray:
    """Envolvente de Hilbert suavizada (20 ms), submuestreada a 100 Hz y estandarizada."""
    env = np.abs(hilbert(audio))
    k = int(0.02 * TARGET_SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")[:: TARGET_SR // 100]
    return (env - env.mean()) / (env.std() + 1e-12)


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Costo DTW normalizado por la longitud del camino."""
    if HAY_LIBROSA:
        D, wp = librosa.sequence.dtw(X=a[None, :], Y=b[None, :], metric="euclidean")
        return float(D[-1, -1] / len(wp))
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1))
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opciones = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(opciones))
            prev = [(i - 1, j - 1), (i - 1, j), (i, j - 1)][k]
            D[i, j] = abs(a[i - 1] - b[j - 1]) + opciones[k]
            L[i, j] = L[prev] + 1
    return float(D[n, m] / L[n, m])


def razon_dtw(sint: list, hold: list, rng) -> tuple:
    n_pares = N_PARES_DTW if HAY_LIBROSA else 150   
    env_s = [envolvente(it["audio"]) for it in sint]
    env_h = [envolvente(it["audio"]) for it in hold]
    pid_h = [it["pid"] for it in hold]

    d_sh = [dtw(env_s[rng.integers(len(env_s))], env_h[rng.integers(len(env_h))])
            for _ in range(n_pares)]

    d_hh, intentos = [], 0
    while len(d_hh) < n_pares and intentos < 50 * n_pares:
        i, j = rng.integers(len(env_h)), rng.integers(len(env_h))
        intentos += 1
        if pid_h[i] != pid_h[j]:
            d_hh.append(dtw(env_h[i], env_h[j]))
    if not d_hh:
        return np.mean(d_sh), np.nan, np.nan
    return np.mean(d_sh), np.mean(d_hh), np.mean(d_sh) / np.mean(d_hh)



# MMD

def _rbf(X, Y, gamma):
    d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.exp(-gamma * np.maximum(d2, 0))


def mmd2(X, Y, gamma) -> float:
    """Estimador de MMD²."""
    m, n = len(X), len(Y)
    Kxx, Kyy, Kxy = _rbf(X, X, gamma), _rbf(Y, Y, gamma), _rbf(X, Y, gamma)
    return float((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
                 + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
                 - 2 * Kxy.mean())


def gamma_mediana(X, Y) -> float:
    Z = np.vstack([X, Y])
    s = (Z ** 2).sum(1)
    d2 = np.maximum(s[:, None] + s[None, :] - 2 * Z @ Z.T, 0)
    med = np.median(d2[np.triu_indices(len(Z), 1)])
    return 1.0 / med if med > 0 else 1.0


def mmd_con_p(X, Y, rng) -> tuple:
    gamma = gamma_mediana(X, Y)
    stat = mmd2(X, Y, gamma)
    Z, m = np.vstack([X, Y]), len(X)
    mayores = 0
    for _ in range(N_PERM_MMD):
        perm = rng.permutation(len(Z))
        if mmd2(Z[perm[:m]], Z[perm[m:]], gamma) >= stat:
            mayores += 1
    return stat, (mayores + 1) / (N_PERM_MMD + 1), gamma



# Memorización

def xcorr_max(items: list, refs: list) -> np.ndarray:
    """Para cada señal, máxima correlación cruzada normalizada con cualquier referencia."""
    L = len(refs[0]["audio"])
    nfft = 2 * L
    R = np.stack([r["audio"] - r["audio"].mean() for r in refs]).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    FR = np.fft.rfft(R, nfft).conj()
    out = []
    for it in items:
        s = (it["audio"] - it["audio"].mean()).astype(np.float32)
        s /= np.linalg.norm(s) + 1e-12
        cc = np.fft.irfft(FR * np.fft.rfft(s, nfft)[None, :], nfft)
        out.append(float(np.abs(cc).max()))
    return np.array(out)



# Principal

def main():
    out_dir = OUT_DIR
    print(f" Train    : {PROCESADOS_DIR / 'train'}")
    print(f" Holdout  : {HOLDOUT_PROC}")
    print(f" Sintético: {SINT_DIR}")
    print(f" Salida   : {out_dir}\n")
    filas_fase = []
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    if not HAY_LIBROSA:
        print(" [AVISO] librosa no está instalado.")

    datos = {}
    for c in CLASES:
        datos[c] = {
            "train": cargar(PROCESADOS_DIR / "train" / c / MODO),
            "hold" : cargar(HOLDOUT_PROC / c / MODO),
            "sint" : cargar(SINT_DIR / c),
        }
        for k, v in datos[c].items():
            if not v:
                raise RuntimeError(f"Sin datos de {k} para {c}. "
                                   "¿Corriste procesar_holdout.py y generate.py?")
        for k in datos[c]:
            for it in datos[c][k]:
                it["logpsd"] = log_psd(it["audio"])
        print(f" Cargado {c:<22} train={len(datos[c]['train'])} "
              f"holdout={len(datos[c]['hold'])} "
              f"({len({i['pid'] for i in datos[c]['hold']})} pac.) "
              f"sint={len(datos[c]['sint'])}")

    ref_hold  = {c: ref_log(datos[c]["hold"]) for c in CLASES}
    ref_train = {c: ref_log(datos[c]["train"]) for c in CLASES}

    filas, filas_bandas = [], []
    M_sint = np.zeros((len(CLASES), len(CLASES)))
    M_real = np.zeros((len(CLASES), len(CLASES)))
    conf_sint = np.zeros((len(CLASES), len(CLASES)), dtype=int)
    conf_real = np.zeros((len(CLASES), len(CLASES)), dtype=int)

    for i, c in enumerate(CLASES):
        print(f"\n Evaluando {c} ...")
        tr, ho, si = datos[c]["train"], datos[c]["hold"], datos[c]["sint"]

        # 1. Correlación espectral y línea base real-real
        corr_s = np.array([pearson(it["logpsd"], ref_hold[c]) for it in si])
        corr_r = []
        for it in ho:
            otros = [o for o in ho if o["pid"] != it["pid"]]
            if otros:
                corr_r.append(pearson(it["logpsd"], ref_log(otros)))
        corr_r = np.array(corr_r) if corr_r else np.array([np.nan])

        # 2. Matriz entre subtipos y centroide más cercano
        for j, c2 in enumerate(CLASES):
            M_sint[i, j] = np.mean([pearson(it["logpsd"], ref_hold[c2]) for it in si])
            M_real[i, j] = (np.nanmean(corr_r) if j == i else
                            np.mean([pearson(it["logpsd"], ref_hold[c2]) for it in ho]))
        for it in si:
            conf_sint[i, np.argmax([pearson(it["logpsd"], ref_train[c2]) for c2 in CLASES])] += 1
        for it in ho:
            conf_real[i, np.argmax([pearson(it["logpsd"], ref_train[c2]) for c2 in CLASES])] += 1

        # 3. DTW
        _, _, r_dtw = razon_dtw(si, ho, rng)

        # 4. MMD
        Xs = np.stack([it["logpsd"] for it in si])
        Xh = np.stack([it["logpsd"] for it in ho])
        Xt = np.stack([it["logpsd"] for it in tr])
        mmd_s, p_s, _ = mmd_con_p(Xs, Xh, rng)
        mmd_b, p_b, _ = mmd_con_p(Xt, Xh, rng)

        # 5. Memorización
        nn_s = xcorr_max(si, tr)
        nn_h = xcorr_max(ho, tr)
        frac_alta = float(np.mean(nn_s > np.percentile(nn_h, 95)))

        # 6. Energía por banda
        for fuente, items in [("train", tr), ("holdout", ho), ("sint", si)]:
            pct = np.array([pct_bandas(it["audio"]) for it in items])
            for j, etq in enumerate(ETIQ_BANDAS):
                filas_bandas.append({"clase": c, "fuente": fuente, "banda_hz": etq,
                                     "media_pct": pct[:, j].mean(),
                                     "std_pct": pct[:, j].std()})

        # 7. Métricas por fase (contraste, intensidad y ubicación del soplo)
        for fuente, items in [("train", tr), ("holdout", ho), ("sint", si)]:
            for it in items:
                if it["mask"].any():
                    filas_fase.append({"clase": c, "fuente": fuente,
                                       **metricas_fase(it["audio"], it["mask"])})

        filas.append({
            "subtipo": c,
            "holdout_pacientes": len({it["pid"] for it in ho}),
            "holdout_segmentos": len(ho),
            "n_sint": len(si),
            "corr_sint_media": corr_s.mean(), "corr_sint_std": corr_s.std(),
            "corr_real_media": np.nanmean(corr_r), "corr_real_std": np.nanstd(corr_r),
            "razon_dtw": r_dtw,
            "mmd2_sint_holdout": mmd_s, "p_mmd_sint": p_s,
            "mmd2_train_holdout": mmd_b, "p_mmd_train": p_b,
            "acierto_centroide_sint": conf_sint[i, i] / max(1, conf_sint[i].sum()),
            "acierto_centroide_real": conf_real[i, i] / max(1, conf_real[i].sum()),
            "xcorr_nn_sint_mediana": float(np.median(nn_s)),
            "xcorr_nn_holdout_mediana": float(np.median(nn_h)),
            "frac_sint_sobre_p95_holdout": frac_alta,
        })

    # Guardar
    df = pd.DataFrame(filas)
    df.to_csv(out_dir / "tabla_III.csv", index=False, float_format="%.4f")
    df_b = pd.DataFrame(filas_bandas)
    df_b.to_csv(out_dir / "energia_bandas.csv", index=False, float_format="%.3f")
    pd.DataFrame(M_sint, index=CLASES, columns=CLASES).to_csv(
        out_dir / "matriz_corr_sint.csv", float_format="%.4f")
    pd.DataFrame(M_real, index=CLASES, columns=CLASES).to_csv(
        out_dir / "matriz_corr_real.csv", float_format="%.4f")
    pd.DataFrame(conf_sint, index=CLASES, columns=CLASES).to_csv(
        out_dir / "confusion_centroide_sint.csv")
    pd.DataFrame(conf_real, index=CLASES, columns=CLASES).to_csv(
        out_dir / "confusion_centroide_real.csv")
    df_f = (pd.DataFrame(filas_fase).groupby(["clase", "fuente"]).median()
            .reindex(pd.MultiIndex.from_product([CLASES, ["train", "holdout", "sint"]])))
    df_f.to_csv(out_dir / "metricas_fase.csv", float_format="%.3f")

    # Figura energía por banda
    fuentes = [("train", "Entrenamiento", "#7F8C8D"), ("holdout", "Holdout", "#2C3E50"),
               ("sint", "Sintético", "#E67E22")]
    fig, axes = plt.subplots(1, len(CLASES), figsize=(15, 3.6), sharey=True)
    xs = np.arange(len(ETIQ_BANDAS))
    for ax, c in zip(axes, CLASES):
        for k, (f, etq, color) in enumerate(fuentes):
            d = df_b[(df_b["clase"] == c) & (df_b["fuente"] == f)]
            ax.bar(xs + (k - 1) * 0.27, d["media_pct"], 0.27, yerr=d["std_pct"],
                   capsize=1.5, error_kw={"elinewidth": 0.6}, color=color, label=etq)
        ax.set_title(c.replace("_", " "), fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(ETIQ_BANDAS, rotation=30, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[0].set_ylabel("% de energía (25–450 Hz)")
    axes[-1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "energia_bandas.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figura matrices de confusión por centroide
    etiquetas = [c.replace("_", "\n") for c in CLASES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, conf, titulo in [(axes[0], conf_real, "Holdout real"),
                             (axes[1], conf_sint, "Sintético")]:
        norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for a in range(len(CLASES)):
            for b in range(len(CLASES)):
                ax.text(b, a, f"{norm[a, b]:.2f}", ha="center", va="center",
                        color="white" if norm[a, b] > 0.5 else "black", fontsize=8)
        ax.set_xticks(range(len(CLASES)))
        ax.set_xticklabels(etiquetas, fontsize=7)
        ax.set_yticks(range(len(CLASES)))
        ax.set_yticklabels(etiquetas, fontsize=7)
        ax.set_xlabel("Subtipo asignado (centroide de entrenamiento)")
        ax.set_title(titulo)
    axes[0].set_ylabel("Subtipo verdadero / solicitado")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_centroide.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Resumen en consola
    cols = ["subtipo", "holdout_pacientes", "n_sint", "corr_sint_media",
            "corr_real_media", "razon_dtw", "mmd2_sint_holdout", "mmd2_train_holdout",
            "acierto_centroide_sint", "acierto_centroide_real",
            "xcorr_nn_sint_mediana", "xcorr_nn_holdout_mediana"]
    print("\n" + df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n Diagonal media de la matriz sintética: {np.mean(np.diag(M_sint)):.3f} "
          f"| fuera de la diagonal: {M_sint[~np.eye(len(CLASES), dtype=bool)].mean():.3f}")
    print(f" Acierto global por centroide: sint {np.trace(conf_sint) / conf_sint.sum():.2f} "
          f"| holdout real {np.trace(conf_real) / conf_real.sum():.2f}")
    print("\n Métricas por fase (medianas):")
    print(df_f.round(2).to_string())
    print(f"\n Resultados en {out_dir}")


if __name__ == "__main__":
    main()
