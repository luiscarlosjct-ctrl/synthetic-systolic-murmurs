import shutil

import numpy as np
import pandas as pd
import soundfile as sf

from comun import (CLASES, EXCEL_FILTRADO, HOLDOUT_CSV, HOLDOUT_PROC, HOLDOUT_RAW,
                   MODO, PROCESADOS_DIR, TARGET_SR, WINDOW_SIZE, aplicar_filtro,
                   leer_wav)


PERCENTIL       = 99.8
MIN_FILAS_CHUNK = 4


def pid_de(nombre: str) -> str:
    return nombre.split("_")[0]


def foco_de(stem: str) -> str:
    """'12345_AV' o '12345_AV_2' → 'AV'."""
    partes = stem.split("_")
    return partes[1].upper() if len(partes) >= 2 else ""


def chunks_desde_tsv(df: pd.DataFrame) -> list:
    chunks, actual = [], []
    for _, row in df.iterrows():
        etiqueta = int(row["label"])
        if etiqueta == 0:
            if actual:
                chunks.append(actual)
                actual = []
        elif not actual:
            if etiqueta == 1:
                actual.append(row)
        else:
            actual.append(row)
    if actual:
        chunks.append(actual)
    return [pd.DataFrame(c) for c in chunks if len(c) >= MIN_FILAS_CHUNK]


def clipping(audio: np.ndarray):
    audio = audio - np.mean(audio)
    p = np.percentile(np.abs(audio), PERCENTIL)
    return None if p == 0 else np.clip(audio, -p, p) / p


def mascara(df: pd.DataFrame, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=int)
    for _, row in df.iterrows():
        s = max(0, int(row["start"] * TARGET_SR))
        e = min(n, int(row["end"] * TARGET_SR))
        if e > s:
            mask[s:e] = int(row["label"])
    return mask


def inicios_s1(mask: np.ndarray) -> list:
    inicios, en_s1 = [], False
    for i, v in enumerate(mask):
        if v == 1 and not en_s1:
            inicios.append(i)
            en_s1 = True
        elif v != 1:
            en_s1 = False
    return inicios


def tsv_segmento(seg_mask: np.ndarray) -> pd.DataFrame:
    filas, actual, inicio = [], seg_mask[0], 0
    for i in range(1, len(seg_mask)):
        if seg_mask[i] != actual:
            filas.append({"start": round(inicio / TARGET_SR, 6),
                          "end": round(i / TARGET_SR, 6), "label": int(actual)})
            actual, inicio = seg_mask[i], i
    filas.append({"start": round(inicio / TARGET_SR, 6),
                  "end": round(len(seg_mask) / TARGET_SR, 6), "label": int(actual)})
    return pd.DataFrame(filas)


def verificar_solapamiento(holdout_ids: set) -> bool:
    tr = {pid_de(p.name) for p in (PROCESADOS_DIR / "train").rglob("*.wav")}
    va = {pid_de(p.name) for p in (PROCESADOS_DIR / "val").rglob("*.wav")}
    ok = True
    print(" Pacientes compartidos entre particiones:")
    for nombre, a, b in [("train ∩ val", tr, va), ("train ∩ holdout", tr, holdout_ids),
                         ("val ∩ holdout", va, holdout_ids)]:
        inter = a & b
        ok &= not inter
        print(f"   {nombre:<16}: {'ninguno' if not inter else sorted(inter)}")
    if not ok:
        print("\n [ATENCIÓN] Hay fuga entre particiones. Borra data/positivos/procesados/ "
              "y vuelve a correr organizar_dataset.py antes de entrenar.")
    return ok


def main():
    df_h = pd.read_csv(HOLDOUT_CSV)
    df_h = df_h[df_h["split"] == "holdout"]
    clase_de = {str(r["patient_id"]): r["clase"] for _, r in df_h.iterrows()}
    verificar_solapamiento(set(clase_de))

    df_x = pd.read_excel(EXCEL_FILTRADO)
    focos_de = {
        str(r["Patient ID"]): (set() if pd.isna(r["Murmur locations"]) else
                               {s.strip().upper() for s in str(r["Murmur locations"]).split("+")})
        for _, r in df_x.iterrows()
    }

    if HOLDOUT_PROC.exists():
        shutil.rmtree(HOLDOUT_PROC)

    conteo = {c: 0 for c in CLASES}
    pacientes = {c: set() for c in CLASES}
    sin_soplo = 0
    contador = {}

    for wav in sorted(HOLDOUT_RAW.glob("*.wav")):
        pid = pid_de(wav.stem)
        clase = clase_de.get(pid)
        if clase not in CLASES:
            continue
        foco = foco_de(wav.stem)
        if foco not in focos_de.get(pid, set()):
            sin_soplo += 1
            continue
        tsv = wav.with_suffix(".tsv")
        if not tsv.exists():
            continue

        audio = leer_wav(wav)
        df_tsv = pd.read_csv(tsv, sep="\t", header=None, names=["start", "end", "label"])
        salida = HOLDOUT_PROC / clase / MODO
        salida.mkdir(parents=True, exist_ok=True)
        clave = f"{pid}_{foco}"
        contador.setdefault(clave, 0)

        for df_c in chunks_desde_tsv(df_tsv):
            t0, t1 = df_c.iloc[0]["start"], df_c.iloc[-1]["end"]
            chunk = audio[round(t0 * TARGET_SR):min(round(t1 * TARGET_SR), len(audio))]
            df_c = df_c.copy()
            df_c["start"] = (df_c["start"] - t0).round(6)
            df_c["end"] = (df_c["end"] - t0).round(6)

            chunk = clipping(chunk)
            if chunk is None or len(chunk) < 24:
                continue
            chunk = aplicar_filtro(chunk)
            m = mascara(df_c, len(chunk))

            ultimo_fin = 0
            for ini in inicios_s1(m):
                if ini < ultimo_fin:
                    continue
                fin = ini + WINDOW_SIZE
                if fin > len(chunk):
                    continue
                seg_mask = m[ini:fin]
                ultimo_fin = fin
                if not {1, 2, 3, 4}.issubset(set(np.unique(seg_mask))):
                    continue
                nombre = f"{clave}_s{contador[clave]:03d}"
                sf.write(salida / f"{nombre}.wav", chunk[ini:fin], TARGET_SR)
                tsv_segmento(seg_mask).to_csv(salida / f"{nombre}.tsv", sep="\t",
                                              header=False, index=False)
                contador[clave] += 1
                conteo[clase] += 1
                pacientes[clase].add(pid)

    print("\n Holdout procesado:")
    for c in CLASES:
        print(f"   {c:<22} {conteo[c]:>4} segmentos | {len(pacientes[c])} pacientes")
    print(f"   Total: {sum(conteo.values())} segmentos")
    print(f"   Grabaciones descartadas por foco sin soplo: {sin_soplo}")
    print(f"\n Guardado en {HOLDOUT_PROC}")


if __name__ == "__main__":
    main()
