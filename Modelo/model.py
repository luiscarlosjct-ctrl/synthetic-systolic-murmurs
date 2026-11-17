import torch
import torch.nn as nn
import math
import torch.nn.functional as F



# Configuración de arquitectura


N_CLASES   = 5     # holosystolic_harsh, holosystolic_blowing,
                   # early_blowing, early_harsh, mid_harsh
N_FASES    = 4     # S1, Sístole, S2, Diástole
IN_CHANNELS = 1 + N_FASES + N_CLASES   # 10 canales



# Time Embedding


class TimeEmbedding(nn.Module):
    """
    Embedding sinusoidal del paso de difusión t.
    
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half    = self.dim // 2
        freqs   = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args    = t[:, None].float() * freqs[None, :]
        emb     = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb



# Class Embedding


class ClassEmbedding(nn.Module):
    """
    Embedding subtipo.
    Recibe el índice de clase (int) y produce un vector denso
    que se inyecta en cada bloque residual de la U-Net,
    igual que time embedding.
    """
    def __init__(self, n_clases: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_clases, dim)

    def forward(self, clase_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(clase_idx)



# MLP proyector de condiciones (time + clase - dim unificado)


class ConditionProjector(nn.Module):
    """
    Proyecta time embedding + class embedding a un vector
    de dimensión time_emb_dim para inyectar en los bloques.
    """
    def __init__(self, base_dim: int, time_emb_dim: int):
        super().__init__()
        self.time_emb  = TimeEmbedding(base_dim)
        self.net = nn.Sequential(
            nn.Linear(base_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor,
                clase_emb: torch.Tensor) -> torch.Tensor:
        t_emb  = self.time_emb(t)           # [B, base_dim]
        cond   = t_emb + clase_emb          # suma — misma dimensión
        return self.net(cond)               # [B, time_emb_dim]



# Bloque residual 1D


class ResBlock1D(nn.Module):
    """
    Bloque residual con inyección de condición (time + subtipo).

    
      GroupNorm → SiLU → Conv → añadir condición → GroupNorm → SiLU → Conv
      + skip connection


    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 groups: int = 8):
        super().__init__()

        # Grupos adaptativos
        def get_groups(channels, max_groups):
            for g in range(max_groups, 0, -1):
                if channels % g == 0:
                    return g
            return 1

        g_in  = get_groups(in_ch,  groups)
        g_out = get_groups(out_ch, groups)

        self.norm1     = nn.GroupNorm(g_in,  in_ch)
        self.conv1     = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2  = nn.GroupNorm(g_out, out_ch)
        self.conv2  = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.act    = nn.SiLU()

        # Skip connection 
        self.skip = (nn.Conv1d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Inyectar condición (time + subtipo) — [B, out_ch] → [B, out_ch, 1]
        h = h + self.cond_proj(cond).unsqueeze(-1)

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)



# Self-Attention 1D


class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention sobre dimensión temporal.
    Se aplica en el bottleneck para captura de dependencias globales.
    Pre-Norm + proyección de salida normalizada.
    """
    def __init__(self, channels: int, n_heads: int = 8,
                 groups: int = 8):
        super().__init__()

        def get_groups(ch, max_g):
            for g in range(max_g, 0, -1):
                if ch % g == 0:
                    return g
            return 1

        g = get_groups(channels, groups)
        self.norm     = nn.GroupNorm(g, channels)
        self.mha      = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads,
            batch_first=True
        )
        self.out_norm = nn.GroupNorm(g, channels)
        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x   = self.norm(x)
        x   = x.permute(0, 2, 1)           # [B, L, C]
        x, _ = self.mha(x, x, x)
        x   = x.permute(0, 2, 1)           # [B, C, L]
        x   = self.out_proj(self.out_norm(x))
        return x + res



# U-Net 1D


class UNet1D(nn.Module):
    """
    U-Net 1D DDPM.

    Entrada: [B, 10, 8192]
      - 1  canal  : audio ruidoso x_t
      - 4  canales: one-hot de estados (S1, Sístole, S2, Diástole)
      - 5  canales: one-hot de subtipo

    Condicionamiento doble:
      - Canales de entrada (concatenados al audio)
      - Embedding de subtipo inyectado en cada bloque residual
        junto con el time embedding

    Salida: [B, 1, 8192] — ruido predicho ε̂

    Arquitectura (4 niveles de downsampling para señal de 8192):
      Encoder: 8192 → 4096 → 2048 → 1024 → 512
      Bottleneck: ResBlock + SelfAttention + ResBlock
      Decoder: 512 → 1024 → 2048 → 4096 → 8192
    """

    def __init__(
        self,
        in_channels  : int = IN_CHANNELS,   # 10
        out_channels : int = 1,
        base_channels: int = 128,
        time_emb_dim : int = 512,
        n_clases     : int = N_CLASES,
        n_heads      : int = 8,
    ):
        super().__init__()

        bc = base_channels   # alias

        # --- Condicionamiento ---
        self.class_emb = ClassEmbedding(n_clases, bc)
        self.cond_proj = ConditionProjector(bc, time_emb_dim)

        # --- Encoder ---
        # Convolución de entrada antes del primer bloque residual
        
        self.in_conv = nn.Conv1d(in_channels, bc, 3, padding=1)
        self.inc     = ResBlock1D(bc,          bc,      time_emb_dim)
        self.down1 = ResBlock1D(bc,          bc * 2,  time_emb_dim)
        self.down2 = ResBlock1D(bc * 2,      bc * 4,  time_emb_dim)
        self.down3 = ResBlock1D(bc * 4,      bc * 8,  time_emb_dim)
        self.down4 = ResBlock1D(bc * 8,      bc * 16, time_emb_dim)

        self.pool  = nn.MaxPool1d(2, 2)

        # --- Bottleneck ---
        self.bot1  = ResBlock1D(bc * 16, bc * 16, time_emb_dim)
        self.attn  = SelfAttention1D(bc * 16, n_heads)
        self.bot2  = ResBlock1D(bc * 16, bc * 16, time_emb_dim)

        # --- Decoder ---
        self.up4   = nn.ConvTranspose1d(bc * 16, bc * 8,  2, stride=2)
        self.dec4  = ResBlock1D(bc * 16, bc * 8,  time_emb_dim)

        self.up3   = nn.ConvTranspose1d(bc * 8,  bc * 4,  2, stride=2)
        self.dec3  = ResBlock1D(bc * 8,  bc * 4,  time_emb_dim)

        self.up2   = nn.ConvTranspose1d(bc * 4,  bc * 2,  2, stride=2)
        self.dec2  = ResBlock1D(bc * 4,  bc * 2,  time_emb_dim)

        self.up1   = nn.ConvTranspose1d(bc * 2,  bc,      2, stride=2)
        self.dec1  = ResBlock1D(bc * 2,  bc,      time_emb_dim)

        # --- Salida ---
        self.outc  = nn.Sequential(
            nn.GroupNorm(8, bc),
            nn.SiLU(),
            nn.Conv1d(bc, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                clase_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : [B, 10, 8192] — audio + estados + subtipos (one-hot)
            t        : [B]            — paso de difusión
            clase_idx: [B]            — índice de subtipo (0-4)
        Returns:
            [B, 1, 8192] — ruido predicho
        """
        # Condición unificada: time + clase
        clase_emb = self.class_emb(clase_idx)       # [B, bc]
        cond      = self.cond_proj(t, clase_emb)    # [B, time_emb_dim]

        # Encoder
        x1 = self.inc(self.in_conv(x),     cond)   # [B, bc,    8192]
        x2 = self.down1(self.pool(x1),     cond)   # [B, bc*2,  4096]
        x3 = self.down2(self.pool(x2),     cond)   # [B, bc*4,  2048]
        x4 = self.down3(self.pool(x3),     cond)   # [B, bc*8,  1024]
        x5 = self.down4(self.pool(x4),     cond)   # [B, bc*16,  512]

        # Bottleneck
        xb = self.bot1(x5,      cond)
        xb = self.attn(xb)
        xb = self.bot2(xb,      cond)

        # Decoder con skip connections
        x  = self.dec4(torch.cat([self.up4(xb), x4], dim=1), cond)
        x  = self.dec3(torch.cat([self.up3(x),  x3], dim=1), cond)
        x  = self.dec2(torch.cat([self.up2(x),  x2], dim=1), cond)
        x  = self.dec1(torch.cat([self.up1(x),  x1], dim=1), cond)

        return self.outc(x)                         # [B, 1, 8192]



# Verificación rápida


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = UNet1D().to(device)

    B = 4
    x         = torch.randn(B, IN_CHANNELS, 8192).to(device)
    t         = torch.randint(0, 1000, (B,)).to(device)
    clase_idx = torch.randint(0, N_CLASES, (B,)).to(device)

    with torch.no_grad():
        out = model(x, t, clase_idx)

    print(f" Entrada : {x.shape}")
    print(f" Salida  : {out.shape}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f" Params  : {n_params:.2f} M")