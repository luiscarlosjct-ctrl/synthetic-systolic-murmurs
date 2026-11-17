import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "Imagenes_informe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SALIDA = OUT_DIR / "arquitectura_unet1d_ablacion.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

C_IN, C_ENC, C_BOT, C_ATT, C_DEC, C_COND = (
    "#E8E8E8", "#9EC5E8", "#F4B183", "#C9A7E0", "#A8D5A2", "#FFE699")
BORDE, GRIS = "#333333", "#777777"
C_CONV = "#D6E6F5"

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(-2.0, 15.8)
ax.set_ylim(-0.1, 10.5)
ax.axis("off")

W, H = 2.3, 0.95


def caja(x, y, texto, color, w=W, h=H, fs=10.5, cond=False):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=color, ec=BORDE, lw=1.1, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs, zorder=4)
    if cond:
        cx, cy = x - w/2 + 0.02, y + h/2 - 0.02
        ax.add_patch(Circle((cx, cy), 0.13, fc=C_COND, ec=BORDE, lw=0.9, zorder=5))
        ax.text(cx, cy, r"$c$", ha="center", va="center", fontsize=9, zorder=6)


def flecha(x0, y0, x1, y1, texto=None, ls="-", color=BORDE,
           tx=0, ty=0, fs=9.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                linestyle=ls, shrinkA=0, shrinkB=0), zorder=2)
    if texto:
        ax.text((x0 + x1)/2 + tx, (y0 + y1)/2 + ty, texto, fontsize=fs,
                ha="center", va="center", color=color, zorder=4)


def rb(c, L):
    return f"ResBlock\n${c} \\times {L}$"


# Codificador / decodificador con conexión de salto
canales = [128, 256, 512, 1024]
longitudes = [8192, 4096, 2048, 1024]
ys = [7.3, 5.8, 4.3, 2.8]
x_enc = [3.4 + 0.75*i for i in range(4)]
x_dec = [11.6 - 0.75*i for i in range(4)]

for i, (c, L, y) in enumerate(zip(canales, longitudes, ys)):
    caja(x_enc[i], y, rb(c, L), C_ENC, cond=True)
    caja(x_dec[i], y, rb(c, L), C_DEC, cond=True)
    flecha(x_enc[i] + W/2, y, x_dec[i] - W/2, y, ls=(0, (4, 3)), color=GRIS)
    ax.text(7.5, y + 0.17, "skip (concat.)", ha="center", fontsize=9,
            color=GRIS, zorder=4)

for i in range(3):
    flecha(x_enc[i], ys[i] - H/2, x_enc[i+1], ys[i+1] + H/2,
           r"MaxPool $\downarrow 2$", tx=-0.85)
    flecha(x_dec[i+1], ys[i+1] + H/2, x_dec[i], ys[i] - H/2,
           r"ConvT $\uparrow 2$", tx=0.75)

# Fila inferior (512 muestras): down4 + cuello de botella.
Y_B, WB = 1.0, 1.9
xs_b = [4.2, 6.4, 8.6, 10.8]
caja(xs_b[0], Y_B, rb(2048, 512), C_ENC, w=WB, fs=10, cond=True)
caja(xs_b[1], Y_B, rb(2048, 512), C_BOT, w=WB, fs=10, cond=True)
caja(xs_b[2], Y_B, "Auto-atención\n8 cabezas", C_ATT, w=WB, fs=10)
caja(xs_b[3], Y_B, rb(2048, 512), C_BOT, w=WB, fs=10, cond=True)
for a, b in zip(xs_b[:-1], xs_b[1:]):
    flecha(a + WB/2, Y_B, b - WB/2, Y_B)
flecha(x_enc[3], ys[3] - H/2, xs_b[0], Y_B + H/2, r"MaxPool $\downarrow 2$", tx=-0.95)
flecha(xs_b[3], Y_B + H/2, x_dec[3], ys[3] - H/2, r"ConvT $\uparrow 2$", tx=0.95)
ax.text(xs_b[2], Y_B - 0.78, "Cuello de botella", ha="center", fontsize=11,
        style="italic")

# Entrada
X_IN, X_CI = -1.1, 0.85
caja(X_IN, 7.3, "Entrada\n$10 \\times 8192$", C_IN, w=1.5, h=1.0, fs=11)
caja(X_CI, 7.3, "Conv$_3$\n$128 \\times 8192$", C_CONV, w=1.4, h=0.8, fs=10)
flecha(X_IN + 0.75, 7.3, X_CI - 0.7, 7.3)
flecha(X_CI + 0.7, 7.3, x_enc[0] - W/2, 7.3)
ax.text(-1.85, 6.2, "Concatenación de canales:", fontsize=9.5, ha="left",
        va="center", weight="bold")
for txt, y in [(r"$x_t$: señal ruidosa (1)", 5.82),
               (r"Fases $S1$, sís., $S2$, diás. (4)", 5.49),
               (r"Subtipo one-hot (5)", 5.16)]:
    ax.text(-1.85, y, txt, fontsize=9.5, ha="left", va="center")

# Detalle de los bloques
ax.text(-1.85, 4.55, "ResBlock:", fontsize=9.5, ha="left", weight="bold")
for txt, y in [(r"GN $\rightarrow$ SiLU $\rightarrow$ Conv$_3$", 4.2),
               (r"$+$ Lineal$(c)$ por canal", 3.87),
               (r"GN $\rightarrow$ SiLU $\rightarrow$ Conv$_3$", 3.54),
               (r"$+$ residual (Conv$_{1}$ si cambia $C$)", 3.21)]:
    ax.text(-1.85, y, txt, fontsize=9.5, ha="left", va="center")
ax.text(-1.85, 2.55, "Auto-atención:", fontsize=9.5, ha="left", weight="bold")
for txt, y in [(r"GN $\rightarrow$ MHA $\rightarrow$ GN", 2.2),
               (r"$\rightarrow$ Conv$_1$ $+$ residual", 1.87)]:
    ax.text(-1.85, y, txt, fontsize=9.5, ha="left", va="center")

# Salida
X_OUT = 14.75
caja(X_OUT, 7.3, "Salida\n$\\hat{\\epsilon}_\\theta$: $1 \\times 8192$", C_IN,
     w=1.8, h=1.0, fs=11)
flecha(x_dec[0] + W/2, 7.3, X_OUT - 0.9, 7.3)
ax.text((x_dec[0] + W/2 + X_OUT - 0.9)/2, 7.72, "GN, SiLU\nConv$_1$",
        ha="center", va="center", fontsize=8.5, color=BORDE)

# Vía de condicionamiento
Y1, Y2, YS = 10.05, 9.15, 9.6
caja(3.3, Y1, "Paso $t$ $\\rightarrow$ emb. sinusoidal $e_t \\in \\mathbb{R}^{128}$",
     C_COND, w=4.6, h=0.62, fs=10)
caja(3.3, Y2, "Subtipo $\\rightarrow$ nn.Embedding $e_y \\in \\mathbb{R}^{128}$",
     C_COND, w=4.6, h=0.62, fs=10)
ax.add_patch(Circle((6.45, YS), 0.25, fc="white", ec=BORDE, lw=1.1, zorder=3))
ax.text(6.45, YS, "+", ha="center", va="center", fontsize=15, zorder=4)
flecha(5.6, Y1, 6.25, YS + 0.12)
flecha(5.6, Y2, 6.25, YS - 0.12)
caja(8.75, YS, "MLP: Lineal $\\rightarrow$ SiLU $\\rightarrow$ Lineal",
     C_COND, w=3.5, h=0.62, fs=10)
flecha(6.7, YS, 7.0, YS)
caja(12.15, YS, "Condición $c \\in \\mathbb{R}^{512}$", C_COND, w=2.4, h=0.62, fs=10.5)
flecha(10.5, YS, 10.95, YS)
ax.text(12.15, YS - 0.6, "inyectada en cada ResBlock", ha="center",
        fontsize=9.5, style="italic")

# Leyenda
leyenda = [
    Line2D([], [], marker="s", ls="", ms=11, mfc=C_CONV, mec=BORDE, label="Conv. de entrada"),
    Line2D([], [], marker="s", ls="", ms=11, mfc=C_ENC, mec=BORDE, label="Codificador"),
    Line2D([], [], marker="s", ls="", ms=11, mfc=C_BOT, mec=BORDE, label="Cuello de botella"),
    Line2D([], [], marker="s", ls="", ms=11, mfc=C_ATT, mec=BORDE, label="Auto-atención"),
    Line2D([], [], marker="s", ls="", ms=11, mfc=C_DEC, mec=BORDE, label="Decodificador"),
    Line2D([], [], marker="o", ls="", ms=11, mfc=C_COND, mec=BORDE,
           label=r"Recibe condición $c$"),
    Line2D([], [], color=GRIS, ls=(0, (4, 3)), label="Conexión de salto"),
]
ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, 0.0),
          fontsize=10, frameon=True, ncol=4)
ax.text(-1.85, 0.05, r"Dimensiones: canales $\times$ muestras", fontsize=9.5,
        va="bottom")

plt.savefig(SALIDA, dpi=300, bbox_inches="tight")
print(f"Guardada: {SALIDA}")