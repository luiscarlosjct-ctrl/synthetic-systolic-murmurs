import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict


# Configuración

VAL_RATIO_POSITIVOS  = 0.15   
VAL_RATIO_NEGATIVOS  = 0.20   
MIN_PACIENTES_VAL    = 2      

CLASES = [
    "holosystolic_harsh",
    "holosystolic_blowing",
    "early_blowing",
    "early_harsh",
    "mid_harsh",
]

MODOS = ["secuencial", "solapado"]



# Utilidades

def extraer_patient_id(filename: str) -> str:
    """'49823_PV_s001.wav' → '49823'"""
    return Path(filename).stem.split("_")[0]


def contar_segmentos_por_paciente(seg_dir: Path) -> dict:
    """
    Cuenta cuántos .wav hay por patient_id en una carpeta.
    Retorna {patient_id: n_segmentos}.
    """
    conteo = defaultdict(int)
    for wav in seg_dir.glob("*.wav"):
        pid = extraer_patient_id(wav.name)
        conteo[pid] += 1
    return dict(conteo)


def seleccionar_val(conteo_por_paciente: dict,
                    val_ratio: float,
                    min_pacientes: int) -> set:
    """
    Selecciona patient_ids para val usando criterio adaptativo:

    - Si con los pacientes más cercanos al promedio se puede alcanzar
      val_ratio sin superar 1.5x val_ratio → usar criterio A (promedio)
    - Si no es posible (clase muy pequeña) → usar criterio B (menor
      a mayor) y registrar advertencia

    Criterio A — más cercano al promedio:
      Ordena pacientes por distancia absoluta al promedio de segmentos.
      Añade hasta cumplir val_ratio y mínimo min_pacientes.
      Evita que pacientes con grabaciones muy largas dominen val.

    Criterio B — menor a mayor (fallback):
      Ordena de menor a mayor segmentos.
      Se usa cuando el dataset es tan pequeño que ningún criterio
      puede cumplir val_ratio sin superar 1.5x val_ratio.

    Retorna set de patient_ids para val.
    """
    total    = sum(conteo_por_paciente.values())
    objetivo = total * val_ratio
    promedio = total / len(conteo_por_paciente) if conteo_por_paciente else 0

    # Criterio A: ordenar por cercanía al promedio
    por_promedio = sorted(
        conteo_por_paciente.items(),
        key=lambda x: abs(x[1] - promedio)
    )

    val_ids_a  = set()
    val_segs_a = 0
    for pid, n_segs in por_promedio:
        val_ids_a.add(pid)
        val_segs_a += n_segs
        if val_segs_a >= objetivo and len(val_ids_a) >= min_pacientes:
            break

    pct_a = val_segs_a / total if total > 0 else 0

    # Si el criterio A no supera 1.5x val_ratio → usarlo
    if pct_a <= val_ratio * 1.5:
        return val_ids_a

    # Criterio B (fallback): menor a mayor segmentos
    print(f"   [INFO] Clase pequena — usando criterio B (menor a mayor)")
    por_menor = sorted(conteo_por_paciente.items(), key=lambda x: x[1])

    val_ids_b  = set()
    val_segs_b = 0
    for pid, n_segs in por_menor:
        val_ids_b.add(pid)
        val_segs_b += n_segs
        if val_segs_b >= objetivo and len(val_ids_b) >= min_pacientes:
            break

    return val_ids_b


def copiar_segmentos(seg_dir: Path, val_ids: set,
                     out_train: Path, out_val: Path) -> tuple:
    """
    Copia .wav y .tsv a train/ o val/ según patient_id.
    Retorna (n_train, n_val).
    """
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0

    for wav in seg_dir.glob("*.wav"):
        pid     = extraer_patient_id(wav.name)
        destino = out_val if pid in val_ids else out_train

        shutil.copy2(wav, destino / wav.name)

        tsv = wav.with_suffix(".tsv")
        if tsv.exists():
            shutil.copy2(tsv, destino / tsv.name)

        if pid in val_ids:
            n_val += 1
        else:
            n_train += 1

    return n_train, n_val



# Pipeline positivos


def organizar_positivos(project_root: Path):

    INPUT_BASE  = project_root / "data" / "positivos" / "segmentados"
    OUTPUT_BASE = project_root / "data" / "positivos" / "procesados"

    print("=" * 65)
    print(f"{'POSITIVOS':^65}")
    print(f"{'Val: ' + str(VAL_RATIO_POSITIVOS*100) + '% de segmentos | min ' + str(MIN_PACIENTES_VAL) + ' pacientes por clase':^65}")
    print("=" * 65)

    registros = []

    for clase in CLASES:
        print(f"\n Clase: {clase}")


        sec_dir = INPUT_BASE / clase / "secuencial"

        if not sec_dir.exists():
            print(f"   [WARN] No existe: {sec_dir.relative_to(project_root)}")
            continue

        conteo = contar_segmentos_por_paciente(sec_dir)
        if not conteo:
            print(f"   [WARN] Sin segmentos")
            continue

        total     = sum(conteo.values())
        val_ids   = seleccionar_val(conteo, VAL_RATIO_POSITIVOS,
                                    MIN_PACIENTES_VAL)
        train_ids = set(conteo.keys()) - val_ids

        val_segs   = sum(conteo[p] for p in val_ids)
        train_segs = sum(conteo[p] for p in train_ids)
        val_pct    = val_segs / total * 100 if total > 0 else 0

        print(f"   Pacientes train : {len(train_ids):>3} "
              f"({train_segs} seg. sec.)")
        print(f"   Pacientes val   : {len(val_ids):>3} "
              f"({val_segs} seg. sec., {val_pct:.1f}%)")
        print(f"   IDs en val      : {sorted(val_ids)}")

        # Copiar ambos modos con los mismos val_ids
        for modo in MODOS:
            modo_dir  = INPUT_BASE / clase / modo
            out_train = OUTPUT_BASE / "train" / clase / modo
            out_val   = OUTPUT_BASE / "val"   / clase / modo

            if not modo_dir.exists():
                continue

            n_tr, n_va = copiar_segmentos(
                modo_dir, val_ids, out_train, out_val
            )
            print(f"   [{modo:<11}] train: {n_tr:>5}  val: {n_va:>4}")

        for pid in sorted(train_ids):
            registros.append({"patient_id": pid, "clase": clase,
                               "split": "train"})
        for pid in sorted(val_ids):
            registros.append({"patient_id": pid, "clase": clase,
                               "split": "val"})

    reg_path = OUTPUT_BASE / "split_positivos.csv"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(reg_path, index=False)
    print(f"\n Registro guardado: {reg_path.relative_to(project_root)}")



# Pipeline negativos


def organizar_negativos(project_root: Path):

    INPUT_BASE  = project_root / "data" / "negativos" / "segmentados"
    OUTPUT_BASE = project_root / "data" / "negativos" / "procesados"

    print("\n" + "=" * 65)
    print(f"{'NEGATIVOS':^65}")
    print(f"{'Val: ' + str(VAL_RATIO_NEGATIVOS*100) + '% de segmentos | min ' + str(MIN_PACIENTES_VAL) + ' pacientes':^65}")
    print("=" * 65)

    registros = []

    sec_dir = INPUT_BASE / "secuencial"
    if not sec_dir.exists():
        print(f" [WARN] No existe: {sec_dir.relative_to(project_root)}")
        return

    conteo = contar_segmentos_por_paciente(sec_dir)
    if not conteo:
        print(f" [WARN] Sin segmentos en secuencial/")
        return

    total     = sum(conteo.values())
    val_ids   = seleccionar_val(conteo, VAL_RATIO_NEGATIVOS,
                                MIN_PACIENTES_VAL)
    train_ids = set(conteo.keys()) - val_ids

    val_segs   = sum(conteo[p] for p in val_ids)
    train_segs = sum(conteo[p] for p in train_ids)
    val_pct    = val_segs / total * 100 if total > 0 else 0

    print(f"\n Pacientes train : {len(train_ids):>4} ({train_segs} seg. sec.)")
    print(f" Pacientes val   : {len(val_ids):>4} "
          f"({val_segs} seg. sec., {val_pct:.1f}%)")

    for modo in MODOS:
        modo_dir  = INPUT_BASE / modo
        out_train = OUTPUT_BASE / "train" / modo
        out_val   = OUTPUT_BASE / "val"   / modo

        if not modo_dir.exists():
            continue

        n_tr, n_va = copiar_segmentos(
            modo_dir, val_ids, out_train, out_val
        )
        print(f" [{modo:<11}] train: {n_tr:>6}  val: {n_va:>5}")

    for pid in sorted(train_ids):
        registros.append({"patient_id": pid, "split": "train"})
    for pid in sorted(val_ids):
        registros.append({"patient_id": pid, "split": "val"})

    reg_path = OUTPUT_BASE / "split_negativos.csv"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(reg_path, index=False)
    print(f"\n Registro guardado: {reg_path.relative_to(project_root)}")



# Función principal


def organizar_dataset():

    FILE_PATH    = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    print(f" Raiz del proyecto : {PROJECT_ROOT}\n")

    organizar_positivos(PROJECT_ROOT)
    organizar_negativos(PROJECT_ROOT)

    print("\n" + "=" * 65)
    print(f"{'ESTRUCTURA DE SALIDA':^65}")
    print("=" * 65)
    print("""
 positivos/procesados/
     split_positivos.csv
     train/
         holosystolic_harsh/secuencial/  *.wav  *.tsv
         holosystolic_harsh/solapado/    *.wav  *.tsv
         holosystolic_blowing/ ...
         early_blowing/ ...
         early_harsh/ ...
         mid_harsh/ ...
     val/
         (misma estructura)

 negativos/procesados/
     split_negativos.csv
     train/
         secuencial/  *.wav  *.tsv
         solapado/    *.wav  *.tsv
     val/
         secuencial/  *.wav  *.tsv
         solapado/    *.wav  *.tsv
    """)


if __name__ == "__main__":
    organizar_dataset()