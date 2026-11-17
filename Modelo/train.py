import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import csv
import soundfile as sf
import numpy as np
import pandas as pd

from model import UNet1D, IN_CHANNELS, N_CLASES
from diffusion import GaussianDiffusion1D, SpectralLoss


# Configuración

MODO           = "secuencial"
WINDOW_SIZE    = 8192
TARGET_SR      = 4000
BATCH_SIZE     = 32
LEARNING_RATE  = 2e-4
EPOCHS         = 1000
TIMESTEPS      = 1000
SAVE_EVERY     = 100       # checkpoint periódico
LAMBDA_SPEC    = 0.1       # peso de la pérdida espectral
GRAD_CLIP      = 1.0       # gradient clipping

CLASES = [
    "holosystolic_harsh",
    "holosystolic_blowing",
    "early_blowing",
    "early_harsh",
    "mid_harsh",
]
CLASE_A_IDX = {c: i for i, c in enumerate(CLASES)}



# Dataset

class PCGDataset(Dataset):

    def __init__(self, procesados_dir: Path, modo: str):
        self.samples = []   # lista de (wav_path, clase_idx)

        for clase in CLASES:
            clase_dir = procesados_dir / "train" / clase / modo
            if not clase_dir.exists():
                continue
            idx = CLASE_A_IDX[clase]
            for wav in sorted(clase_dir.glob("*.wav")):
                self.samples.append((wav, idx))

        print(f" Dataset cargado: {len(self.samples)} segmentos")
        for clase in CLASES:
            n = sum(1 for _, idx in self.samples
                    if idx == CLASE_A_IDX[clase])
            print(f"   {clase:<25}: {n}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        wav_path, clase_idx = self.samples[i]
        tsv_path = wav_path.with_suffix(".tsv")

        # Audio
        audio, _ = sf.read(wav_path)
        audio    = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        # [1, 8192]

        # Máscara de estados
        mask = torch.zeros(WINDOW_SIZE, dtype=torch.long)
        if tsv_path.exists():
            df = pd.read_csv(tsv_path, sep="\t", header=None,
                             names=["start", "end", "label"])
            for _, row in df.iterrows():
                s = max(0, int(row["start"] * TARGET_SR))
                e = min(WINDOW_SIZE, int(row["end"] * TARGET_SR))
                if e > s:
                    mask[s:e] = int(row["label"])

        mask = mask.unsqueeze(0)   # [1, 8192]

        return audio, mask, torch.tensor(clase_idx, dtype=torch.long)



# Dataset de validación

class PCGValDataset(Dataset):
    

    def __init__(self, procesados_dir: Path, modo: str):
        self.samples = []
        for clase in CLASES:
            clase_dir = procesados_dir / "val" / clase / modo
            if not clase_dir.exists():
                continue
            idx = CLASE_A_IDX[clase]
            for wav in sorted(clase_dir.glob("*.wav")):
                self.samples.append((wav, idx))
        print(f" Val dataset   : {len(self.samples)} segmentos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        wav_path, clase_idx = self.samples[i]
        tsv_path = wav_path.with_suffix(".tsv")
        audio, _ = sf.read(wav_path)
        audio    = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        mask     = torch.zeros(WINDOW_SIZE, dtype=torch.long)
        if tsv_path.exists():
            df = pd.read_csv(tsv_path, sep="\t", header=None,
                             names=["start", "end", "label"])
            for _, row in df.iterrows():
                s = max(0, int(row["start"] * TARGET_SR))
                e = min(WINDOW_SIZE, int(row["end"] * TARGET_SR))
                if e > s:
                    mask[s:e] = int(row["label"])
        mask = mask.unsqueeze(0)
        return audio, mask, torch.tensor(clase_idx, dtype=torch.long)



# Entrenamiento


def train():

    FILE_PATH    = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent

    PROCESADOS_DIR = PROJECT_ROOT / "data" / "positivos" / "procesados"
    CKPT_DIR       = PROJECT_ROOT / "checkpoints" / "ddpm_ablacion_inconv"   
    CSV_PATH       = CKPT_DIR / "historial_loss.csv"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Modo          : {MODO.upper()}")
    print(f" Device        : {device.upper()}")
    print(f" Batch size    : {BATCH_SIZE}")
    print(f" Epochs        : {EPOCHS}")
    print(f" Timesteps     : {TIMESTEPS}")
    print(f" LR            : {LEARNING_RATE}")
    print(f" Lambda spec   : {LAMBDA_SPEC}\n")

    # --- Datasets ---
    train_ds = PCGDataset(PROCESADOS_DIR, MODO)
    val_ds   = PCGValDataset(PROCESADOS_DIR, MODO)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          shuffle=False, drop_last=False,
                          num_workers=2, pin_memory=True)

    # --- Modelo ---
    model     = UNet1D().to(device)
    diffusion = GaussianDiffusion1D(timesteps=TIMESTEPS, device=device)
    spectral  = SpectralLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f" Parametros    : {n_params:.2f} M\n")

    # Reanudar entrenamiento si existe checkpoint periódico
    start_epoch  = 1
    best_val_loss = float("inf")
    ckpt_periodico = CKPT_DIR / f"ckpt_{MODO}_periodico.pt"
    ckpt_mejor     = CKPT_DIR / f"ckpt_{MODO}_mejor.pt"

    if ckpt_periodico.exists():
        print(f" Reanudando desde checkpoint: {ckpt_periodico.name}")
        ckpt = torch.load(ckpt_periodico, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f" Reanudando desde época {start_epoch}\n")
    else:
        # Iniciar CSV
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    # Bucle de entrenamiento
    print(" Iniciando entrenamiento...\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Ep {epoch}/{EPOCHS} [train]",
                    leave=False)
        for audio, mask, clase_idx in pbar:
            audio     = audio.to(device)
            mask      = mask.to(device)
            clase_idx = clase_idx.to(device)

           
            t = torch.randint(0, TIMESTEPS,
                              (audio.shape[0],), device=device).long()

            loss = diffusion.get_loss(
                model, audio, mask, clase_idx, t,
                spectral, LAMBDA_SPEC
            )

            optimizer.zero_grad()
            loss.backward()

            
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss / len(train_dl)
        scheduler.step()
        lr_actual = optimizer.param_groups[0]["lr"]

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio, mask, clase_idx in val_dl:
                audio     = audio.to(device)
                mask      = mask.to(device)
                clase_idx = clase_idx.to(device)
                t = torch.randint(0, TIMESTEPS,
                                  (audio.shape[0],), device=device).long()
                loss = diffusion.get_loss(
                    model, audio, mask, clase_idx, t,
                    spectral, LAMBDA_SPEC
                )
                val_loss += loss.item()

        avg_val = val_loss / len(val_dl) if len(val_dl) > 0 else 0.0

        print(f" Ep {epoch:>4}/{EPOCHS} | "
              f"Train: {avg_train:.6f} | "
              f"Val: {avg_val:.6f} | "
              f"LR: {lr_actual:.2e}")

        # Guardar historial
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, avg_train, avg_val, lr_actual]
            )

        # Checkpoint periódico 
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            torch.save({
                "epoch"         : epoch,
                "model"         : model.state_dict(),
                "optimizer"     : optimizer.state_dict(),
                "scheduler"     : scheduler.state_dict(),
                "train_loss"    : avg_train,
                "val_loss"      : avg_val,
                "best_val_loss" : best_val_loss,
            }, ckpt_periodico)
            print(f"   Checkpoint periódico guardado: {ckpt_periodico.name}")

        # Mejor modelo 
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch"    : epoch,
                "model"    : model.state_dict(),
                "val_loss" : avg_val,
            }, ckpt_mejor)
            print(f"   Mejor modelo guardado: val_loss={avg_val:.6f}")

    print(f"\n Entrenamiento completado.")
    print(f" Mejor val loss: {best_val_loss:.6f}")
    print(f" Checkpoints en: {CKPT_DIR.relative_to(PROJECT_ROOT)}/")
    


if __name__ == "__main__":
    train()