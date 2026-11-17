import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram
from pathlib import Path


def graficar_dataset_raw():

    # RUTAS 
    FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    TRAIN_DIR = PROJECT_ROOT / "data" / "Dataset sintetico ablacion" / "holosystolic_harsh"
    OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "Imagenes_informe"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ESTILO LATEX
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    # SELECCIÓN DE ARCHIVOS
    ARCHIVOS_A_GRAFICAR = [
        "sint_abl_holosystolic_harsh_s0001_bpm104.wav",
    ]

    # PARÁMETROS DEL ESPECTROGRAMA
    NPERSEG = 128      # 32 ms a 4 kHz
    NOVERLAP = 96      # 75 % de solapamiento
    VMIN_DB = -80      # Piso del rango dinámico - dB relativos al máximo
    RIBBON_ALTO = 0.07 # Fracción superior del espectrograma

    print(f"Buscando los archivos en: {TRAIN_DIR.relative_to(PROJECT_ROOT)}...")

    todos_los_wav = list(TRAIN_DIR.rglob("*.wav"))
    wavs_seleccionados = [wav for wav in todos_los_wav if wav.name in ARCHIVOS_A_GRAFICAR]

    if not wavs_seleccionados:
        print("Error: No se encontró ninguno archivo.")
        return

    

    # MAPA DE COLORES 
    color_map = {
        0: ('#D3D3D3', '(Ruido/No clasificado)'),
        1: ('#FF9999', r'$S_1$'),
        2: ('#FFFF99', 'Sístole'),
        3: ('#99CCFF', r'$S_2$'),
        4: ('#99FF99', 'Diástole')
    }

    print(f"Gráficas en {OUT_DIR.relative_to(PROJECT_ROOT)}/")

    for wav_path in wavs_seleccionados:
        tsv_path = wav_path.with_suffix('.tsv')

        if not tsv_path.exists():
            print(f"No se encontró el .tsv para {wav_path.name}.")
            continue

        
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        duracion = len(audio) / sr
        time_axis = np.arange(len(audio)) / sr  

        
        df_tsv = pd.read_csv(tsv_path, sep='\t', header=None, names=['start', 'end', 'label'])

        
        f_s, t_s, Sxx = spectrogram(audio, fs=sr, nperseg=NPERSEG,
                                    noverlap=NOVERLAP, scaling="density")
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        Sxx_db -= Sxx_db.max()  # 0 dB = máximo del registro

        
        fig = plt.figure(figsize=(15, 7))
        gs = GridSpec(2, 2, figure=fig,
                      width_ratios=[1, 0.015],
                      height_ratios=[1, 1.2],
                      hspace=0.08, wspace=0.02)

        ax = fig.add_subplot(gs[0, 0])
        ax_spec = fig.add_subplot(gs[1, 0], sharex=ax)
        cax = fig.add_subplot(gs[1, 1])

        
        ax.plot(time_axis, audio, color='#2C3E50', linewidth=0.3, label='Audio')

        
        im = ax_spec.pcolormesh(t_s, f_s, Sxx_db, shading="gouraud",
                                cmap="inferno", vmin=VMIN_DB, vmax=0)

        # SEGMENTACIÓN
        etiquetas_usadas = set()
        for _, row in df_tsv.iterrows():
            start_t, end_t, label = row['start'], row['end'], int(row['label'])
            if label not in color_map:
                continue
            color, label_name = color_map[label]

            leyenda = label_name if label not in etiquetas_usadas else None
            etiquetas_usadas.add(label)
            ax.axvspan(start_t, end_t, color=color, alpha=0.4, label=leyenda)

            
            ax_spec.axvspan(start_t, end_t, ymin=0.0, ymax=RIBBON_ALTO,
                            color=color, alpha=0.95, linewidth=0)

        
        ax.set_title(f"Archivo de audio sintético {wav_path.name}")
        ax.set_ylabel(r"Amplitud")
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(labelbottom=False)  

        ax_spec.set_xlim([0, duracion])  
        
        f_max = sr / 2
        h_ribbon = f_max * RIBBON_ALTO / (1 - RIBBON_ALTO)
        ax_spec.set_ylim([-h_ribbon, f_max])
        ax_spec.axhline(0, color='black', linewidth=0.8)  

        
        ticks = [t for t in ax_spec.get_yticks() if 0 <= t <= f_max]
        ax_spec.set_yticks(ticks)
        ax_spec.set_xlabel(r"Tiempo $(\mathrm{s})$")
        ax_spec.set_ylabel(r"Frecuencia $(\mathrm{Hz})$")

        fig.colorbar(im, cax=cax, label=r"Potencia relativa $(\mathrm{dB})$")

        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        # EXPORTAR
        ruta_salida = OUT_DIR / f"raw_grafica_{wav_path.stem}.png"
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generada: {wav_path.name}")

    print("\nProceso completado")


if __name__ == "__main__":
    graficar_dataset_raw()