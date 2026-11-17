from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf



# RUTAS


PROJECT_ROOT   = Path("..").resolve()          
TRAIN_DIR      = PROJECT_ROOT / "data" / "positivos" / "procesados" / "train"  
HOLDOUT_DIR    = PROJECT_ROOT / "data" / "holdout" / "procesados"               
SINT_DIR       = PROJECT_ROOT / "data" / "Dataset sintetico ablacion"                 
OUT_DIR        = PROJECT_ROOT / "outputs" / "evaluacion_ablacion"



# Config


SR = 4000

CLASES = ["early_blowing", "early_harsh", "mid_harsh",
          "holosystolic_blowing", "holosystolic_harsh"]
TITULOS = {"early_blowing": "Early systolic blowing", "early_harsh": "Early systolic harsh",
           "mid_harsh": "Midsystolic harsh", "holosystolic_blowing": "Holosystolic blowing",
           "holosystolic_harsh": "Holosystolic harsh"}
FUENTES = [("train", "Entrenamiento", "#1f77b4", "o"),
           ("holdout", "Holdout", "#ff7f0e", "s"),
           ("sint", "Sintético", "#2ca02c", "D")]


def mascara(tsv: Path, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=np.int64)
    df = pd.read_csv(tsv, sep="\t", header=None, names=["start", "end", "label"])
    for _, r in df.iterrows():
        s, e = max(0, int(r["start"] * SR)), min(n, int(r["end"] * SR))
        if e > s:
            m[s:e] = int(r["label"])
    return m


def tercios(audio: np.ndarray, mask: np.ndarray):
    """Fracción de la potencia sistólica en cada tercio de la sístole"""
    p = audio ** 2
    tramos = np.flatnonzero(np.diff(np.r_[0, (mask == 2).astype(int), 0])).reshape(-1, 2)
    t = np.zeros(3)
    for a, b in tramos:
        if b - a < 3:
            continue
        c = np.linspace(a, b, 4).astype(int)
        t += [p[c[k]:c[k + 1]].mean() for k in range(3)]
    return t / t.sum() if t.sum() > 0 else None


def cargar(carpeta: Path) -> np.ndarray:
    filas = []
    for w in sorted(carpeta.glob("*.wav")):
        tsv = w.with_suffix(".tsv")
        if not tsv.exists():
            continue
        x, _ = sf.read(w)
        x = x[:, 0] if x.ndim > 1 else x
        r = tercios(x.astype(np.float64), mascara(tsv, len(x)))
        if r is not None:
            filas.append(r)
    return np.array(filas).reshape(-1, 3)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datos = {c: {"train": cargar(TRAIN_DIR / c / "secuencial"),
                 "holdout": cargar(HOLDOUT_DIR / c / "secuencial"),
                 "sint": cargar(SINT_DIR / c)} for c in CLASES}

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(CLASES), figsize=(7.2, 2.3), sharey=True)
    x = np.arange(3)
    desplaz = {"train": -0.12, "holdout": 0.0, "sint": 0.12}
    filas = []

    for ax, c in zip(axes, CLASES):
        ax.axhline(1 / 3, color="#B0B0B0", lw=0.8, ls="--", zorder=0,
                   label="Reparto uniforme (1/3)")
        for f, etq, color, marca in FUENTES:
            arr = datos[c][f]
            if not len(arr):
                continue
            med = np.median(arr, axis=0)
            q1, q3 = np.percentile(arr, [25, 75], axis=0)
            xs = x + desplaz[f]
            ax.errorbar(xs, med, yerr=[med - q1, q3 - med], color=color, marker=marca,
                        ms=4, lw=1.5, elinewidth=0.8, capsize=2,
                        label=f"{etq}")
            for k in range(3):
                filas.append({"clase": c, "fuente": f, "tercio": k + 1, "n": len(arr),
                              "mediana": med[k], "q1": q1[k], "q3": q3[k]})
        ax.set_title(TITULOS[c], fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Inicio", "Centro", "Final"])
        ax.set_xlim(-0.4, 2.4)
        ax.grid(True, axis="y", color="#E5E5E5", lw=0.6)
    axes[0].set_ylabel("Fracción de la energía sistólica")
    axes[0].set_ylim(0, 0.75)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.07), fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT_DIR / f"tercios_sistole.{ext}", dpi=300, bbox_inches="tight")
    plt.show()

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / "tercios_sistole.csv", index=False, float_format="%.3f")
    print(df.pivot_table(index=["clase", "fuente"], columns="tercio",
                         values="mediana").round(2).to_string())
    print(f"\n Guardado en {OUT_DIR}")


if __name__ == "__main__":
    main()