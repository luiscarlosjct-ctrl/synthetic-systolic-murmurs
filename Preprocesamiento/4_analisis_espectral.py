import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import soundfile as sf
import pandas as pd
from pathlib import Path
from scipy.signal import welch, spectrogram



# Estilo LaTeX



plt.rcParams.update({
    "text.usetex"             : False,
    "mathtext.fontset"        : "cm",
    "font.family"             : "STIXGeneral",   # STIX = serif estilo LaTeX
    "font.serif"              : ["STIXGeneral", "DejaVu Serif", "serif"],
    "axes.titlesize"          : 12,
    "axes.labelsize"          : 11,
    "xtick.labelsize"         : 9,
    "ytick.labelsize"         : 9,
    "legend.fontsize"         : 9,
    "legend.title_fontsize"   : 9,
    "figure.dpi"              : 150,
    "axes.spines.top"         : False,
    "axes.spines.right"       : False,
})



# Configuración


VENTANA_S  = 3.0        # segundos a visualizar en espectrogramas
N_MUESTRAS = None       # None = todos los archivos de cada clase (PSD y energía)


UMBRAL_CONV   = 0.30    # 30% de separación relativa
VENTANA_CONV  = 5       # bins de confirmación

CLASES = [
    "holosystolic_harsh",
    "holosystolic_blowing",
    "early_blowing",
    "early_harsh",
    "mid_harsh",
]


# Selección manual de archivos para los espectrogramas (Figura 1)

ARCHIVOS_ESPECTROGRAMA = {
    "holosystolic_harsh"   : "49823_PV.wav",
    "holosystolic_blowing" : "29045_TV.wav",
    "early_blowing"        : "50054_MV.wav",
    "early_harsh"          : "68306_PV.wav",
    "mid_harsh"            : "49627_TV.wav",
}

COLOR_CLASE = {
    "holosystolic_harsh"   : "#E24B4A",
    "holosystolic_blowing" : "#378ADD",
    "early_blowing"        : "#1D9E75",
    "early_harsh"          : "#EF9F27",
    "mid_harsh"            : "#8E44AD",
}

FASE_COLOR = {1: "#EE3333", 2: "#DDAA00", 3: "#3366EE", 4: "#33AA33"}
FASE_LABEL = {
    1: r"$S_1$",
    2: r"S\'{i}stole",
    3: r"$S_2$",
    4: r"Di\'{a}stole",
}
RIBBON_FRAC = 0.045     



# Utilidades


def leer_audio_y_tsv(wav_path):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float64)
    tsv_path = wav_path.with_suffix(".tsv")
    df = pd.read_csv(tsv_path, sep="\t", header=None,
                     names=["start", "end", "label"])
    return audio, sr, df


def encontrar_ventana_estable(df_tsv, audio_len, sr, duracion=VENTANA_S):

    duracion_total = audio_len / sr
    s1_starts = [float(r["start"]) for _, r in df_tsv.iterrows()
                 if int(r["label"]) == 1]

    if not s1_starts:
        return 0.0, min(duracion, duracion_total)

    t_c0 = duracion_total * 0.33
    t_c1 = duracion_total * 0.67

    # Candidatos en tercio central con ventana completa
    candidatos = [t for t in s1_starts
                  if t_c0 <= t <= t_c1 and t + duracion <= duracion_total]

    if not candidatos:
        # Fallback: cualquier S1 con ventana completa
        candidatos = [t for t in s1_starts if t + duracion <= duracion_total]

    if not candidatos:
        # Grabación más corta que duracion — tomar desde el primer S1
        t_ini = s1_starts[0]
        return t_ini, duracion_total

    centro  = duracion_total / 2
    t_ini   = min(candidatos, key=lambda t: abs(t - centro))
    return t_ini, t_ini + duracion


def listar_archivos(input_dir, clase):
    """Todos los .wav de la clase (ordenados), o los primeros N_MUESTRAS."""
    archivos = sorted((input_dir / clase).glob("*.wav"))
    return archivos if N_MUESTRAS is None else archivos[:N_MUESTRAS]


_CACHE_PSD = {}

def psds_de_clase(input_dir, clase):
    """PSD de la señal completa de cada archivo de la clase.
    Se calcula una sola vez y se reutiliza en las figuras 2 y 4."""
    clave = (str(input_dir), clase)
    if clave not in _CACHE_PSD:
        freqs, psds, sr = None, [], None
        for wav_path in listar_archivos(input_dir, clase):
            audio, sr, _ = leer_audio_y_tsv(wav_path)
            freqs, psd   = calcular_psd(audio, sr)
            psds.append(psd)
        _CACHE_PSD[clave] = (freqs, np.array(psds), sr)
    return _CACHE_PSD[clave]


def calcular_psd(audio, sr, nperseg=256):
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg,
                       noverlap=nperseg // 2, scaling="density")
    return freqs, psd


def extraer_fase(audio, sr, df_tsv, label):
    segs = []
    for _, row in df_tsv.iterrows():
        if int(row["label"]) == label:
            s = max(0, int(row["start"] * sr))
            e = min(len(audio), int(row["end"] * sr))
            if e > s:
                segs.append(audio[s:e])
    return np.concatenate(segs) if segs else np.array([])


def detectar_convergencia(freqs, curvas_dict):
    """
    Encuentra el primer bin donde la separación relativa entre curvas
    cae bajo UMBRAL_CONV y se mantiene sin superar 2×UMBRAL_CONV
    en los siguientes VENTANA_CONV bins.

    Retorna la frecuencia de convergencia en Hz, o None si no converge.
    """
    claves = list(curvas_dict.keys())
    matriz = np.array([curvas_dict[k] for k in claves])   # [n, n_freqs]
    n_freqs = matriz.shape[1]

    for i in range(n_freqs - VENTANA_CONV):
        maxs = np.max(matriz[:, i], axis=0)
        mins = np.min(matriz[:, i], axis=0)

        if maxs == 0:
            continue

        sep = (maxs - mins) / maxs

        if sep < UMBRAL_CONV:
            # Verificar que se mantiene
            bloque = matriz[:, i:i + VENTANA_CONV]
            maxs_b = np.max(bloque, axis=0)
            mins_b = np.min(bloque, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                seps_b = np.where(maxs_b > 0,
                                  (maxs_b - mins_b) / maxs_b, 0)
            if np.all(seps_b < 2 * UMBRAL_CONV):
                return freqs[i]

    return None


def agregar_ribbon_y_franjas(ax, df_tsv, t_ini, t_fin, y_min, y_max):
    ribbon_h = (y_max - y_min) * RIBBON_FRAC
    usadas   = set()

    for _, row in df_tsv.iterrows():
        label = int(row["label"])
        if label not in FASE_COLOR:
            continue
        s = max(float(row["start"]), t_ini) - t_ini
        e = min(float(row["end"]),   t_fin) - t_ini
        if e <= s:
            continue
        color = FASE_COLOR[label]

        # Franja semitransparente
        ax.axvspan(s, e, color=color, alpha=0.18, zorder=0)

        # Ribbon sólido inferior
        ax.fill_between([s, e], [y_min, y_min],
                        [y_min + ribbon_h, y_min + ribbon_h],
                        color=color, alpha=1.0, zorder=3)
        usadas.add(label)

    return usadas



# Figura 1 — Espectrogramas (3 s, ventana etable)


def figura_espectrogramas(input_dir, output_dir):

    fig, axes = plt.subplots(len(CLASES), 1,
                             figsize=(13, 3.6 * len(CLASES)))
    fig.suptitle(
        r"Espectrogramas por clase $-$ ventana de $3\,\mathrm{s}$ ",
        fontsize=13, y=1.005
    )

    for ax, clase in zip(axes, CLASES):
        nombre = ARCHIVOS_ESPECTROGRAMA.get(clase, "")

        # Buscar el archivo directamente en la carpeta de clase
        wav_path = None
        candidato = input_dir / clase / nombre
        if candidato.exists():
            wav_path = candidato

        if wav_path is None:
            ax.set_title(
                f"{clase.replace('_', ' ')}  -  "
                f"'{nombre}' no encontrado",
                fontsize=10
            )
            ax.axis("off")
            continue
        audio, sr, df = leer_audio_y_tsv(wav_path)

        t_ini, t_fin = encontrar_ventana_estable(df, len(audio), sr)
        dur_real = t_fin - t_ini
        m_ini    = int(t_ini * sr)
        m_fin    = int(t_fin * sr)
        seg      = audio[m_ini:m_fin]

        # Espectrograma
        f_s, t_s, Sxx = spectrogram(seg, fs=sr, nperseg=128,
                                     noverlap=96, scaling="density")
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        im = ax.pcolormesh(t_s, f_s, Sxx_db, shading="gouraud",
                           cmap="inferno", vmin=-80, vmax=0)

        # Ribbon + franjas
        y_min, y_max = 0, sr // 2
        agregar_ribbon_y_franjas(ax, df, t_ini, t_fin, y_min, y_max)

        ax.set_xlim([0, dur_real])
        ax.set_ylim([y_min, y_max])
        ax.set_ylabel(r"Frecuencia $(\mathrm{Hz})$")
        ax.set_xlabel(r"Tiempo $(\mathrm{s})$")
        ax.set_title(
            f"{clase.replace('_', ' ')}  ---  {wav_path.name}",
            fontsize=10
        )
        plt.colorbar(im, ax=ax, label=r"$\mathrm{dB}$")

    # Leyenda de fases
    patches = [
        mpatches.Patch(color=FASE_COLOR[l],
                       label=f"$S_1$" if l == 1 else
                             f"$S_2$" if l == 3 else
                             "Sistole" if l == 2 else "Diastole")
        for l in [1, 2, 3, 4]
    ]
    fig.legend(handles=patches, loc="lower left", ncol=4,
               bbox_to_anchor=(0.5, -0.015), fontsize=10,
               title="Fase cardiaca", title_fontsize=10,
               frameon=True)

    plt.tight_layout()
    out = output_dir / "01_espectrogramas_por_clase.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Guardado: {out.name}")



# Figura 2 — PSD promedio por subtipo con zona de convergencia


def figura_psd_por_clase(input_dir, output_dir):

    fig, ax = plt.subplots(figsize=(12, 6))
    curvas_medias = {}
    freqs_ref = None

    for clase in CLASES:
        freqs, psds, sr = psds_de_clase(input_dir, clase)
        if len(psds) == 0:
            continue
        if freqs_ref is None:
            freqs_ref = freqs
        print(f"   PSD {clase:<22}: {len(psds)} archivos")

        media = np.mean(psds, axis=0)
        std   = np.std(psds,  axis=0)
        curvas_medias[clase] = media

        ax.semilogy(freqs_ref, media,
                    color=COLOR_CLASE[clase], linewidth=1.8,
                    label=clase.replace("_", " "))
        ax.fill_between(freqs_ref,
                        np.maximum(media - std, 1e-12),
                        media + std,
                        color=COLOR_CLASE[clase], alpha=0.15)


    ax.set_xlim([0, sr // 2])
    ax.set_xlabel(r"Frecuencia $(\mathrm{Hz})$")
    ax.set_ylabel(r"$\mathrm{PSD}\ \left(\mathrm{V}^2/\mathrm{Hz}\right)$")
    ax.set_title(
        "Densidad espectral de potencia promedio por clase\n"
        #r"Zona gris = región de convergencia entre clases"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = output_dir / "02_psd_por_clase.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Guardado: {out.name}")



# Figura 3 — PSD Sístole vs Diástole con zona de convergencia


def figura_psd_por_fase(input_dir, output_dir):

    fig, axes = plt.subplots(1, len(CLASES),
                             figsize=(5 * len(CLASES), 5),
                             sharey=True)
    fig.suptitle(
        r"$\mathrm{PSD}$ por fase cardiaca: Sistole vs. Diastole" + "\n",
        #r"Zona gris $=$ frecuencia a partir de la cual ambas fases convergen",
        fontsize=12
    )

    for ax, clase in zip(axes, CLASES):
        archivos = listar_archivos(input_dir, clase)
        if not archivos:
            ax.set_title(clase.replace("_", "\n"), fontsize=9)
            continue

        psd_sis, psd_dia = [], []
        freqs_ref = None

        for wav_path in archivos:
            audio, sr, df = leer_audio_y_tsv(wav_path)
            seg_s = extraer_fase(audio, sr, df, label=2)
            seg_d = extraer_fase(audio, sr, df, label=4)

            if len(seg_s) > 64:
                f, p = calcular_psd(seg_s, sr)
                psd_sis.append(p)
                if freqs_ref is None:
                    freqs_ref = f
            if len(seg_d) > 64:
                f, p = calcular_psd(seg_d, sr)
                psd_dia.append(p)

        media_s = np.mean(psd_sis, axis=0) if psd_sis else None
        media_d = np.mean(psd_dia, axis=0) if psd_dia else None

        if media_s is not None:
            ax.semilogy(freqs_ref, media_s,
                        color="#E24B4A", linewidth=1.8, label="Sistole")
        if media_d is not None:
            ax.semilogy(freqs_ref, media_d,
                        color="#378ADD", linewidth=1.8, label="Diastole")


        ax.set_xlim([0, sr // 2])
        ax.set_xlabel(r"Frecuencia $(\mathrm{Hz})$", fontsize=9)
        ax.set_title(clase.replace("_", "\n"), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel(
        r"$\mathrm{PSD}\ \left(\mathrm{V}^2/\mathrm{Hz}\right)$",
        fontsize=10
    )
    plt.tight_layout()
    out = output_dir / "03_psd_sistole_vs_diastole.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Guardado: {out.name}")



# Figura 4 — Energía por banda


def figura_energia_acumulada(input_dir, output_dir):

    bandas    = [(0, 25), (25, 100), (100, 200),
                 (200, 400), (400, 600), (600, 1000)]
    etiquetas = [r"$0$-$25$", r"$25$-$100$", r"$100$-$200$",
                 r"$200$-$400$", r"$400$-$600$", r"$600$-$1000$"]

    fig, ax = plt.subplots(figsize=(13, 6))
    x     = np.arange(len(bandas))
    width = 0.15

    for i, clase in enumerate(CLASES):
        freqs, psds, sr = psds_de_clase(input_dir, clase)
        if len(psds) == 0:
            continue

        energia_por_banda = []
        for banda_low, banda_high in bandas:
            mask = (freqs >= banda_low) & (freqs < banda_high)
            energia_por_banda.append(np.mean(np.sum(psds[:, mask], axis=1)))

        total = sum(energia_por_banda) or 1
        pct   = [e / total * 100 for e in energia_por_banda]

        ax.bar(x + i * width, pct, width,
               label=clase.replace("_", " "),
               color=COLOR_CLASE[clase], alpha=0.85)

    ax.set_xticks(x + width * (len(CLASES) - 1) / 2)
    ax.set_xticklabels(
        [e + r"$\ \mathrm{Hz}$" for e in etiquetas],
        fontsize=9
    )
    ax.set_ylabel(r"$\%$ de energía total")
    ax.set_xlabel(r"Banda de frecuencia")
    ax.set_title(
        "Distribucion de energia por banda de frecuencia y clase"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    out = output_dir / "04_energia_por_banda.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Guardado: {out.name}")



# Ejecución


def analisis_espectral():

    FILE_PATH    = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    INPUT_DIR  = PROJECT_ROOT / "data" / "positivos" / "raw_filtrado" #"raw_clipping"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analisis_espectral_filtrado" #"analisis_espectral"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f" Fuente  : {INPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print(f" Salida  : {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print(f" Ventana : {VENTANA_S} s (zona central, inicio en S1)")
    print(f" Umbral convergencia : {UMBRAL_CONV * 100:.0f}%")
    print(f" Ventana confirmacion: {VENTANA_CONV} bins\n")

    figura_espectrogramas(INPUT_DIR, OUTPUT_DIR)
    figura_psd_por_clase(INPUT_DIR, OUTPUT_DIR)
    figura_psd_por_fase(INPUT_DIR, OUTPUT_DIR)
    figura_energia_acumulada(INPUT_DIR, OUTPUT_DIR)

    print("\n Analisis completado.")
    

if __name__ == "__main__":
    analisis_espectral()