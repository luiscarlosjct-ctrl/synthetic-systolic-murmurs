import random
import shutil
import pandas as pd
from pathlib import Path



# Configuración

HOLDOUT_RATIO_POSITIVOS = 0.15   # 15% de pacientes positivos
HOLDOUT_RATIO_NEGATIVOS = 0.15   # 15% de pacientes negativos
RANDOM_SEED             = 42


CLASS_MAP = {
    ("holosystolic",   "harsh")  : "holosystolic_harsh",
    ("holosystolic",   "blowing"): "holosystolic_blowing",
    ("early-systolic", "blowing"): "early_blowing",
    ("early-systolic", "harsh")  : "early_harsh",
    ("mid-systolic",   "harsh")  : "mid_harsh",
}



# Utilidades


def _get_clase(row) -> str | None:
    """Determina la subclase clínica de un paciente positivo."""
    timing  = str(row.get("Systolic murmur timing",  "")).strip().lower()
    quality = str(row.get("Systolic murmur quality", "")).strip().lower()
    return CLASS_MAP.get((timing, quality), None)


def _split_estratificado(ids: list, ratio: float, seed: int) -> tuple:
    """
    Divide una lista de IDs en holdout y trabajo de forma estratificada.
    Retorna (holdout_ids, trabajo_ids).
    """
    ids_sorted = sorted(ids)
    random.seed(seed)
    random.shuffle(ids_sorted)
    n_holdout = max(1, round(len(ids_sorted) * ratio))
    return set(ids_sorted[:n_holdout]), set(ids_sorted[n_holdout:])


def _mover_paciente(patient_id: str, raw_dir: Path,
                    destino_dir: Path) -> int:
    """
    Mueve todos los .wav y .tsv de un paciente desde raw_dir a destino_dir.
    Retorna el número de archivos movidos.
    """
    archivos = list(raw_dir.rglob(f"{patient_id}_*.*"))
    movidos  = 0
    for archivo in archivos:
        if archivo.suffix in (".wav", ".tsv"):
            destino = destino_dir / archivo.name
            if not destino.exists():
                shutil.move(str(archivo), str(destino))
            movidos += 1
    return movidos



# Función principal


def holdout_split():

    FILE_PATH    = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    RAW_DIR    = PROJECT_ROOT / "data" / "raw"
    EXCEL_PATH = RAW_DIR / "training_data_filtrada.xlsx"
    CSV_PATH   = next(RAW_DIR.rglob("training_data.csv"), None)

    # Carpetas de holdout
    HOLDOUT_POS = PROJECT_ROOT / "data" / "holdout" / "positivos"
    HOLDOUT_NEG = PROJECT_ROOT / "data" / "holdout" / "negativos"
    HOLDOUT_POS.mkdir(parents=True, exist_ok=True)
    HOLDOUT_NEG.mkdir(parents=True, exist_ok=True)

    # Registro de IDs movidos
    REGISTRO_DIR = PROJECT_ROOT / "data" / "holdout"
    REGISTRO_DIR.mkdir(parents=True, exist_ok=True)




    # 1. Cargar metadatos


    if not EXCEL_PATH.exists():
        print(f" Error: No se encontro {EXCEL_PATH}")
        return
    if CSV_PATH is None:
        print(f" Error: No se encontro training_data.csv en {RAW_DIR}")
        return

    df_excel = pd.read_excel(EXCEL_PATH)
    df_csv   = pd.read_csv(CSV_PATH)

    # IDs de positivos desde el Excel filtrado
    ids_positivos = set(df_excel["Patient ID"].astype(str))

    # IDs de negativos
    mask_neg = (
        (df_csv["Murmur"]   == "Absent") &
        (df_csv["Outcome"]  == "Normal")
    )
    ids_negativos = set(df_csv[mask_neg]["Patient ID"].astype(str))
    ids_negativos -= ids_positivos   # sanity check

    print(f" Pacientes positivos (Present+Abnormal) : {len(ids_positivos)}")
    print(f" Pacientes negativos (Absent+Normal)    : {len(ids_negativos)}")


    # 2. Split estratificado por subtipo


    ids_por_clase = {}
    for _, row in df_excel.iterrows():
        pid   = str(row["Patient ID"])
        clase = _get_clase(row)
        if clase:
            ids_por_clase.setdefault(clase, []).append(pid)

    holdout_pos_ids = set()
    trabajo_pos_ids = set()

    print("\n Split estratificado POSITIVOS:")
    print(f"   {'Clase':<25} {'Holdout':>8} {'Trabajo':>8}  {'Total':>6}")
    

    registros_pos = []
    for clase, ids in sorted(ids_por_clase.items()):
        h_ids, t_ids = _split_estratificado(ids, HOLDOUT_RATIO_POSITIVOS,
                                            RANDOM_SEED)
        holdout_pos_ids |= h_ids
        trabajo_pos_ids |= t_ids
        print(f"   {clase:<25} {len(h_ids):>8} {len(t_ids):>8}  {len(ids):>6}")
        for pid in h_ids:
            registros_pos.append({"patient_id": pid, "clase": clase,
                                  "split": "holdout"})
        for pid in t_ids:
            registros_pos.append({"patient_id": pid, "clase": clase,
                                  "split": "trabajo"})

    print(f"   {'TOTAL':<25} {len(holdout_pos_ids):>8} "
          f"{len(trabajo_pos_ids):>8}  {len(ids_positivos):>6}")


    # 3. Split de NEGATIVOS


    holdout_neg_ids, trabajo_neg_ids = _split_estratificado(
        list(ids_negativos), HOLDOUT_RATIO_NEGATIVOS, RANDOM_SEED
    )

    print(f"\n Split NEGATIVOS:")
    print(f"   Holdout : {len(holdout_neg_ids)}")
    print(f"   Trabajo : {len(trabajo_neg_ids)}")
    print(f"   Total   : {len(ids_negativos)}")

    registros_neg = [
        {"patient_id": pid, "clase": "negativo", "split": "holdout"}
        for pid in holdout_neg_ids
    ] + [
        {"patient_id": pid, "clase": "negativo", "split": "trabajo"}
        for pid in trabajo_neg_ids
    ]


    # 4. Sanity check — sin solapamiento entre positivos y negativos


    overlap = holdout_pos_ids & holdout_neg_ids
    if overlap:
        print(f"\n ERROR: {len(overlap)} IDs aparecen en ambos holdouts: {overlap}")
        return


    # 5. Mover archivos al holdout


    print("\n Moviendo archivos al holdout...")

    total_movidos_pos = 0
    for pid in sorted(holdout_pos_ids):
        movidos = _mover_paciente(pid, RAW_DIR, HOLDOUT_POS)
        total_movidos_pos += movidos

    total_movidos_neg = 0
    for pid in sorted(holdout_neg_ids):
        movidos = _mover_paciente(pid, RAW_DIR, HOLDOUT_NEG)
        total_movidos_neg += movidos

    print(f"   Archivos positivos movidos : {total_movidos_pos}")
    print(f"   Archivos negativos movidos : {total_movidos_neg}")


    # 6. Guardar registros CSV


    reg_pos_path = REGISTRO_DIR / "holdout_positivos.csv"
    reg_neg_path = REGISTRO_DIR / "holdout_negativos.csv"

    pd.DataFrame(registros_pos).to_csv(reg_pos_path, index=False)
    pd.DataFrame(registros_neg).to_csv(reg_neg_path, index=False)

    print(f"\n Registros guardados:")
    print(f"   {reg_pos_path.relative_to(PROJECT_ROOT)}")
    print(f"   {reg_neg_path.relative_to(PROJECT_ROOT)}")


    # 7. Resumen


    print(f"\n Holdout completado.")
   
    print(f"   data/holdout/positivos/ - {len(holdout_pos_ids)} pacientes")
    print(f"   data/holdout/negativos/ - {len(holdout_neg_ids)} pacientes")
    print(f"   data/raw/               - {len(trabajo_pos_ids) + len(trabajo_neg_ids)} pacientes")
 


if __name__ == "__main__":
    holdout_split()