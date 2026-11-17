from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


#  RUTAS
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("..").resolve()
FUENTES_DIR = {
    "train"  : lambda c: PROJECT_ROOT / "data" / "positivos" / "procesados" / "train" / c / "secuencial",
    "holdout": lambda c: PROJECT_ROOT / "data" / "holdout" / "procesados" / c / "secuencial",
    "sint"   : lambda c: PROJECT_ROOT / "data" / "Dataset sintetico ablacion" / c,
}
OUT_DIR = PROJECT_ROOT / "outputs" / "figuras_informe"


SR = 4000
CLASES = ["early_blowing", "early_harsh", "mid_harsh",
          "holosystolic_blowing", "holosystolic_harsh"]
TITULOS = {"early_blowing": "Early systolic blowing", "early_harsh": "Early systolic harsh",
           "mid_harsh": "Midsystolic harsh", "holosystolic_blowing": "Holosystolic blowing",
           "holosystolic_harsh": "Holosystolic harsh"}
FASES = {1: "$S1$", 2: "Sístole", 3: "$S2$", 4: "Diástole"}
ESTILO = [("train", "Entrenamiento", "#1f77b4"),
          ("holdout", "Holdout", "#ff7f0e"),
          ("sint", "Sintético", "#2ca02c")]


def mascara(tsv: Path, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=int)
    for s, e, l in pd.read_csv(tsv, sep="\t", header=None).values:
        m[max(0, int(s * SR)):min(n, int(e * SR))] = int(l)
    return m


def potencia_relativa(carpeta: Path) -> np.ndarray:
    
    filas = []
    for w in sorted(carpeta.glob("*.wav")):
        tsv = w.with_suffix(".tsv")
        if not tsv.exists():
            continue
        x, _ = sf.read(w)
        x = (x[:, 0] if x.ndim > 1 else x).astype(np.float64)
        m = mascara(tsv, len(x))
        p, total = x ** 2, (x ** 2).mean()
        if total <= 0 or not all((m == f).any() for f in FASES):
            continue
        filas.append([p[m == f].mean() / total for f in FASES])
    return np.array(filas).reshape(-1, 4)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datos = {c: {f: potencia_relativa(d(c)) for f, d in FUENTES_DIR.items()} for c in CLASES}

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(CLASES), figsize=(7.2, 2.4), sharey=True)
    x = np.arange(len(FASES))
    ancho = 0.26
    filas = []
    for ax, c in zip(axes, CLASES):
        for k, (f, etq, color) in enumerate(ESTILO):
            arr = datos[c][f]
            if not len(arr):
                continue
            med = np.median(arr, axis=0)
            q1, q3 = np.percentile(arr, [25, 75], axis=0)
            ax.bar(x + (k - 1) * ancho, med, ancho, color=color, label=etq,
                   yerr=[med - q1, q3 - med], error_kw={"elinewidth": 0.6, "capsize": 1.5})
            for j, nombre in enumerate(FASES.values()):
                filas.append({"clase": c, "fuente": f, "fase": nombre, "n": len(arr),
                              "mediana": med[j], "q1": q1[j], "q3": q3[j]})
        ax.set_yscale("log")
        ax.set_title(TITULOS[c], fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(FASES.values()), rotation=35, fontsize=7)
        ax.grid(True, axis="y", which="major", color="#E5E5E5", lw=0.6)
        ax.axhline(1, color="#B0B0B0", lw=0.8, ls="--", zorder=0)
    axes[0].set_ylabel("Potencia relativa al segmento")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=7)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT_DIR / f"potencia_estados.{ext}", dpi=300, bbox_inches="tight")
    plt.show()

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / "potencia_estados.csv", index=False, float_format="%.3f")
    print(df.pivot_table(index=["clase", "fuente"], columns="fase",
                         values="mediana").round(2).to_string())
    print(f"\n Guardado en {OUT_DIR}")


if __name__ == "__main__":
    main()