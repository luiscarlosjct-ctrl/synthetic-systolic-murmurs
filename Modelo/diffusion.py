import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# Pérdida espectral


class SpectralLoss(nn.Module):
    """
    Pérdida en el dominio de la frecuencia usando STFT.
    Penaliza diferencias en la estructura frecuencial del ruido predicho,
    complementando la L1 que opera en el dominio temporal.

    PCG de 8192 muestras a 4000 Hz:
      - n_fft=256 → resolución frecuencial de ~15.6 Hz
      - hop_length=64 → resolución temporal de ~16 ms
    """
    def __init__(self, n_fft: int = 256, hop_length: int = 64):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, 1, L] — ruido predicho y real
        """
        B = pred.shape[0]
        p = pred.squeeze(1)     # [B, L]
        t = target.squeeze(1)   # [B, L]

        # STFT
        window = torch.hann_window(self.n_fft, device=pred.device)
        sp = torch.stft(p, self.n_fft, self.hop_length,
                        window=window, return_complex=True)
        st = torch.stft(t, self.n_fft, self.hop_length,
                        window=window, return_complex=True)

        # Magnitud log — más estable que magnitud lineal
        mag_p = torch.log1p(sp.abs())
        mag_t = torch.log1p(st.abs())

        return F.l1_loss(mag_p, mag_t)



# Proceso de difusión gaussiana 1D


class GaussianDiffusion1D:
    """
    Controlador matemático del proceso DDPM para señales 1D.

    """

    def __init__(self, timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float   = 0.02,
                 device: str       = "cpu"):

        self.timesteps = timesteps
        self.device    = device

        # Schedule lineal de betas (Ho et al. 2020)
        self.betas = torch.linspace(beta_start, beta_end,
                                    timesteps).to(device)
        self.alphas           = 1.0 - self.betas
        self.alphas_cumprod   = torch.cumprod(self.alphas, dim=0)

        # α̅_{t-1} — desplazado un paso (α̅_0 = 1)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )

        # Raíces para el forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )

        # FIX 1: varianza posterior β̃_t (ec. 7 de Ho et al. 2020)
        # β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        # Clip para evitar log(0) en t=0
        self.posterior_variance_clipped = torch.clamp(
            self.posterior_variance, min=1e-20
        )
        self.sqrt_posterior_variance = torch.sqrt(
            self.posterior_variance_clipped
        )

    
    # Forward process: x_0 → x_t (reparametrización)
    

    def forward_diffusion_sample(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple:
        """
        Aplica ruido gaussiano en un solo paso usando 
        reparametrización: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

        Args:
            x_0: [B, 1, L] — señal limpia
            t  : [B]        — pasos de difusión
        Returns:
            (x_t, noise) — señal ruidosa y ruido aplicado
        """
        noise            = torch.randn_like(x_0)
        sqrt_alpha_hat   = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus   = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus * noise
        return x_t, noise

    
    # Función de pérdida: L1 temporal + L1 espectral
    

    def get_loss(self, model: nn.Module,
                 x_0      : torch.Tensor,
                 mask     : torch.Tensor,
                 clase_idx: torch.Tensor,
                 t        : torch.Tensor,
                 spectral_loss_fn: SpectralLoss,
                 lambda_spectral : float = 0.1) -> torch.Tensor:
        """
        Calcula la pérdida combinada L1 + espectral.

        Args:
            model          : U-Net 1D
            x_0            : [B, 1, L]  — audio limpio
            mask           : [B, L]     — máscara de estados (labels 1-4)
            clase_idx      : [B]        — índice de subtipo (0-4)
            t              : [B]        — pasos de difusión (incluye t=0)
            spectral_loss_fn: instancia de SpectralLoss
            lambda_spectral: peso de la pérdida espectral

        Returns:
            loss total escalar
        """
        x_noisy, true_noise = self.forward_diffusion_sample(x_0, t)

        # One-hot de estados: mask [B, L] → [B, 4, L]
        mask_sq    = mask.squeeze(1).long()        # [B, L]
        # Labels 
        mask_sq    = torch.clamp(mask_sq - 1, 0, 3)
        mask_onehot = F.one_hot(mask_sq, num_classes=4).float()
        mask_onehot = mask_onehot.permute(0, 2, 1)  # [B, 4, L]

        # One-hot subtipo: [B] → [B, 5, L]
        clase_onehot = F.one_hot(clase_idx, num_classes=5).float()
        clase_onehot = clase_onehot.unsqueeze(-1).expand(
            -1, -1, x_0.shape[-1]
        )                                           # [B, 5, L]

        # Concatenar: [B, 10, L]
        model_input = torch.cat(
            [x_noisy, mask_onehot, clase_onehot], dim=1
        )

        # Predicción del ruido
        pred_noise = model(model_input, t, clase_idx)

        # Pérdida L1 temporal
        loss_l1 = F.l1_loss(true_noise, pred_noise)

        # Pérdida espectral
        loss_spec = spectral_loss_fn(pred_noise, true_noise)

        return loss_l1 + lambda_spectral * loss_spec

   
    # Paso inverso: x_T → x_0 (generación)
    

    @torch.no_grad()
    def sample(self, model        : nn.Module,
               mask_condition     : torch.Tensor,
               clase_idx          : torch.Tensor,
               n_samples          : int,
               seq_length         : int) -> torch.Tensor:
        """
        Proceso inverso genera señales desde ruido.

        
        Args:
            model          : U-Net entrenada
            mask_condition : [B, 1, L] — máscaras de estados como condición
            clase_idx      : [B]       — índice de subtipo
            n_samples      : número de muestras a generar
            seq_length     : longitud de la señal (8192)

        Returns:
            [B, 1, L] — señales sintéticas generadas
        """
        model.eval()

        # One-hot de estados
        mask_sq     = mask_condition.squeeze(1).long()
        mask_sq     = torch.clamp(mask_sq - 1, 0, 3)
        mask_onehot = F.one_hot(mask_sq, num_classes=4).float()
        mask_onehot = mask_onehot.permute(0, 2, 1)   # [B, 4, L]

        # One-hot de subtipo
        clase_onehot = F.one_hot(clase_idx, num_classes=5).float()
        clase_onehot = clase_onehot.unsqueeze(-1).expand(
            -1, -1, seq_length
        )                                             # [B, 5, L]

        # Iniciar desde ruido puro x_T ~ N(0, I)
        x = torch.randn((n_samples, 1, seq_length)).to(self.device)

        for i in reversed(range(self.timesteps)):
            t_batch = torch.full((n_samples,), i,
                                 device=self.device, dtype=torch.long)

            model_input = torch.cat(
                [x, mask_onehot, clase_onehot], dim=1
            )
            pred_noise = model(model_input, t_batch, clase_idx)

            # Coeficientes del paso inverso
            alpha      = self.alphas[i]
            alpha_hat  = self.alphas_cumprod[i]
            beta       = self.betas[i]

            # Media posterior 
            x = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - alpha_hat)) * pred_noise
            )

            # Añadir varianza posterior β̃_t 
            if i > 0:
                noise = torch.randn_like(x)
                x = x + self.sqrt_posterior_variance[i] * noise

        model.train()
        return x



# Verificación rápida


if __name__ == "__main__":
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = GaussianDiffusion1D(timesteps=1000, device=device)

    B, L = 4, 8192
    x_0       = torch.randn(B, 1, L).to(device)
    t         = torch.randint(0, 1000, (B,)).to(device)   
    mask      = torch.ones(B, 1, L, dtype=torch.long).to(device)
    clase_idx = torch.randint(0, 5, (B,)).to(device)

    x_t, noise = diffusion.forward_diffusion_sample(x_0, t)
    print(f" Forward process: x_0{tuple(x_0.shape)} → x_t{tuple(x_t.shape)}")

    # Verificar que posterior_variance
    pv = diffusion.posterior_variance
    print(f" Posterior variance: min={pv.min():.6f}  max={pv.max():.6f}")
    print(f" (Si min~0 y max~beta_end → correcto)")