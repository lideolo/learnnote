""" 
RhythmMamba: Fast Remote Physiological Measurement with Arbitrary Length Videos
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba

class Fusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5,dim=24):
        super(Fusion_Stem, self).__init__()


        self.stem11 = nn.Sequential(nn.Conv2d(3, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(12, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.apha*x_path1 + self.belta*x_path2

        return x
    

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class ROIAwareFrameStem(nn.Module):
    def __init__(self, dim, roi_count=5):
        super().__init__()
        self.roi_count = roi_count
        self.token_stem = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.quality_gate = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def _roi_boxes(self, device):
        boxes = torch.tensor([
            [0.08, 0.22, 0.22, 0.78],  # forehead
            [0.34, 0.02, 0.72, 0.42],  # left cheek in cropped face coordinates
            [0.34, 0.58, 0.72, 0.98],  # right cheek
            [0.24, 0.30, 0.76, 0.70],  # nose / mid-face
            [0.66, 0.22, 0.96, 0.78],  # chin / lower face
            [0.16, 0.12, 0.84, 0.88],  # broad central face fallback
        ], dtype=torch.float32, device=device)
        return boxes[:self.roi_count]

    def _crop_mean(self, x, box):
        _, _, _, H, W = x.shape
        y1 = int(torch.floor(box[0] * H).item())
        x1 = int(torch.floor(box[1] * W).item())
        y2 = int(torch.ceil(box[2] * H).item())
        x2 = int(torch.ceil(box[3] * W).item())
        y1 = max(0, min(y1, H - 1))
        x1 = max(0, min(x1, W - 1))
        y2 = max(y1 + 1, min(y2, H))
        x2 = max(x1 + 1, min(x2, W))
        return x[:, :, :, y1:y2, x1:x2].mean(dim=(3, 4))

    def _quality_features(self, roi_tokens):
        roi_mean = roi_tokens.mean(dim=-1)
        motion = torch.mean(torch.abs(roi_mean[:, 1:] - roi_mean[:, :-1]), dim=1)
        stability = torch.std(roi_mean, dim=1, unbiased=False)
        centered = roi_mean - roi_mean.mean(dim=1, keepdim=True)
        spectrum = torch.abs(torch.fft.rfft(centered, dim=1, norm="ortho"))
        sharpness = spectrum[:, 1:].amax(dim=1) / (spectrum[:, 1:].sum(dim=1) + 1e-6)
        features = torch.stack([motion, stability, sharpness], dim=-1)
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (features - mean) / std

    def forward(self, x):
        roi_tokens = []
        for box in self._roi_boxes(x.device):
            roi_tokens.append(self._crop_mean(x, box).permute(0, 2, 1))
        roi_tokens = torch.stack(roi_tokens, dim=2)
        roi_tokens = roi_tokens + 0.1 * self.token_stem(roi_tokens)
        quality_features = self._quality_features(roi_tokens)
        roi_logits = self.quality_gate(quality_features).squeeze(-1)
        roi_weights = torch.softmax(roi_logits, dim=-1)
        fused_tokens = torch.sum(roi_tokens * roi_weights[:, None, :, None], dim=2)
        return fused_tokens, roi_tokens, roi_weights, quality_features


class AdaptiveSpectralGate(nn.Module):
    def __init__(
        self,
        dim,
        hr_low=0.7,
        hr_high=2.5,
        prior_strength=0.2,
        adaptive_strength=0.1,
    ):
        super().__init__()
        hidden_dim = max(8, dim // 4)
        self.hr_low = hr_low
        self.hr_high = hr_high
        self.prior_strength = prior_strength
        self.adaptive_strength = adaptive_strength
        self.adaptive_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def _prior(self, freqs):
        prior = torch.full_like(freqs, 0.5)
        hr_band = (freqs >= self.hr_low) & (freqs <= self.hr_high)
        harmonic_band = (freqs >= 2 * self.hr_low) & (freqs <= 2 * self.hr_high)
        low_drift = freqs < 0.5
        high_noise = freqs > 5.0
        prior = torch.where(hr_band, torch.ones_like(prior), prior)
        prior = torch.where(harmonic_band & (~hr_band), torch.full_like(prior, 0.75), prior)
        prior = torch.where(low_drift, torch.full_like(prior, 0.25), prior)
        prior = torch.where(high_noise, torch.full_like(prior, 0.35), prior)
        return prior

    def forward(self, roi_tokens, token_fs):
        T = roi_tokens.shape[1]
        freqs = torch.fft.rfftfreq(T, d=1.0 / token_fs).to(roi_tokens.device)
        x_fft = torch.fft.rfft(roi_tokens, dim=1, norm="ortho")
        prior = self._prior(freqs).view(1, -1, 1, 1)
        learned = torch.sigmoid(self.adaptive_gate(torch.log1p(torch.abs(x_fft))))
        gate = 1.0 + self.prior_strength * (prior - 0.5)
        gate = gate + self.adaptive_strength * (learned - 0.5)
        gate = torch.clamp(gate, 0.5, 1.5)
        gated_tokens = torch.fft.irfft(x_fft * gate, n=T, dim=1, norm="ortho")
        return gated_tokens, gate.mean(dim=-1)


class PeriodicTokenModulator(nn.Module):
    def __init__(
        self,
        dim,
        hr_low_bpm=45,
        hr_high_bpm=150,
        fourier_bands=(0.7, 1.0, 1.3, 1.7, 2.1, 2.5),
        use_pe=True,
        use_hr_modulation=True,
    ):
        super().__init__()
        self.use_pe = use_pe
        self.use_hr_modulation = use_hr_modulation
        self.register_buffer("fourier_bands", torch.tensor(fourier_bands, dtype=torch.float32))
        self.register_buffer("hr_bpm", torch.arange(hr_low_bpm, hr_high_bpm + 1, dtype=torch.float32))
        self.pe_proj = nn.Linear(len(fourier_bands) * 2, dim)
        self.hr_head = nn.Linear(dim, hr_high_bpm - hr_low_bpm + 1)
        self.phase_proj = nn.Linear(4, dim)

    def forward(self, x, token_fs):
        B, T, _ = x.shape
        time = torch.arange(T, dtype=x.dtype, device=x.device) / token_fs
        aux = {}

        if self.use_pe:
            phase = 2 * math.pi * time[:, None] * self.fourier_bands.to(dtype=x.dtype, device=x.device)[None, :]
            pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
            x = x + self.pe_proj(pe).unsqueeze(0)

        pooled = x.mean(dim=1)
        hr_logits = self.hr_head(pooled)
        aux["hr_logits"] = hr_logits

        if self.use_hr_modulation:
            hr_prob = torch.softmax(hr_logits, dim=-1)
            expected_hz = torch.sum(hr_prob * self.hr_bpm.to(dtype=x.dtype, device=x.device), dim=-1) / 60.0
            phase = 2 * math.pi * expected_hz[:, None] * time[None, :]
            phase_features = torch.stack(
                [torch.sin(phase), torch.cos(phase), torch.sin(2 * phase), torch.cos(2 * phase)],
                dim=-1,
            )
            x = x + self.phase_proj(phase_features)

        return x, aux


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim),
        )


    def forward(self, x):
        B, N, C = x.shape
  
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho') # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)

        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba


class Block_mamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        mamba_d_state=48,
        mamba_d_conv=4,
        mamba_expand=2,
        multi_temporal_paths=3,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(
            dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.mlp = Frequencydomain_FFN(dim,mlp_ratio)
        self.multi_temporal_paths = multi_temporal_paths
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, D, C = x.size()
        #Multi-temporal Parallelization
        path = self.multi_temporal_paths
        segment = 2**(path-1)
        if D % segment != 0:
            raise ValueError(f"Temporal length {D} must be divisible by segment {segment}.")
        tt = D // segment
        x_r = x.repeat(segment,1,1)
        x_o = x_r.clone()
        for i in range(1,segment):
            x_o[i*B:(i+1)*B,:D-i*tt,:] = x_r[i*B:(i+1)*B,i*tt:,:]
        x_o = self.attn(x_o)
        for i in range(1,segment):
            for j in range(i):
                x_o[0:B, tt*i: tt*(i+1) , :] = x_o[0:B, tt*i: tt*(i+1) , :] + x_o[B*(j+1):B*(j+2), tt*(i-j-1): tt*(i-j) , :]
            x_o[0:B, tt*i: tt*(i+1) , :] = x_o[0:B, tt*i: tt*(i+1) , :] / (i+1)
        x = x + self.drop_path(self.norm1(x_o[0:B]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class RhythmMamba(nn.Module):
    def __init__(self, 
                 depth=24, 
                 embed_dim=96, 
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 mamba_d_state=48,
                 mamba_d_conv=4,
                 mamba_expand=2,
                 multi_temporal_paths=3,
                 use_roi_stem=False,
                 roi_count=5,
                 roi_residual_scale=1.0,
                 use_spectral_gate=False,
                 use_periodic_pe=False,
                 use_periodic_modulation=False,
                 return_aux=False,
                 sampling_rate=30.0,
                 spectral_hr_low=0.7,
                 spectral_hr_high=2.5,
                 spectral_prior_strength=0.2,
                 spectral_adaptive_strength=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.embed_dim = embed_dim
        self.use_roi_stem = use_roi_stem
        self.use_spectral_gate = use_spectral_gate
        self.use_periodic = use_periodic_pe or use_periodic_modulation
        self.return_aux = return_aux
        self.sampling_rate = sampling_rate
        self.roi_residual_scale = roi_residual_scale

        self.Fusion_Stem = Fusion_Stem(dim=embed_dim//4)
        self.attn_mask = Attention_mask()

        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim//4, embed_dim, kernel_size=(2, 5, 5), stride=(2, 1, 1),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim),
        )

        if self.use_roi_stem:
            self.roi_stem = ROIAwareFrameStem(embed_dim, roi_count=roi_count)
        if self.use_spectral_gate:
            self.spectral_gate = AdaptiveSpectralGate(
                embed_dim,
                hr_low=spectral_hr_low,
                hr_high=spectral_hr_high,
                prior_strength=spectral_prior_strength,
                adaptive_strength=spectral_adaptive_strength,
            )
        if self.use_periodic:
            self.periodic_modulator = PeriodicTokenModulator(
                embed_dim,
                use_pe=use_periodic_pe,
                use_hr_modulation=use_periodic_modulation,
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([Block_mamba(
            dim = embed_dim, 
            mlp_ratio = mlp_ratio,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            multi_temporal_paths=multi_temporal_paths,
            drop_path=inter_dpr[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(depth)])

        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1,stride=1, padding=0)

        # init
        self.apply(segm_init_weights)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self._init_v110_parameters()

    def _init_v110_parameters(self):
        if hasattr(self, "roi_stem"):
            nn.init.zeros_(self.roi_stem.quality_gate[-1].weight)
            nn.init.zeros_(self.roi_stem.quality_gate[-1].bias)
        if hasattr(self, "spectral_gate"):
            nn.init.zeros_(self.spectral_gate.adaptive_gate[-1].weight)
            nn.init.zeros_(self.spectral_gate.adaptive_gate[-1].bias)
        if hasattr(self, "periodic_modulator"):
            nn.init.zeros_(self.periodic_modulator.pe_proj.weight)
            nn.init.zeros_(self.periodic_modulator.pe_proj.bias)
            nn.init.zeros_(self.periodic_modulator.phase_proj.weight)
            nn.init.zeros_(self.periodic_modulator.phase_proj.bias)

    def forward(self, x):
        B, D, C, H, W = x.shape

        x = self.Fusion_Stem(x)    #[N*D C H/8 W/8]
        x = x.view(B,D,self.embed_dim//4,H//8,W//8).permute(0,2,1,3,4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        token_fs = self.sampling_rate * x.shape[2] / D
        aux = {}
        if self.use_roi_stem:
            global_tokens = torch.mean(torch.mean(x, 4), 3)
            global_tokens = rearrange(global_tokens, 'b c t -> b t c')
            roi_fused, roi_tokens, roi_weights, quality_features = self.roi_stem(x)
            if self.use_spectral_gate:
                roi_tokens, spectral_gate = self.spectral_gate(roi_tokens, token_fs)
                roi_fused = torch.sum(roi_tokens * roi_weights[:, None, :, None], dim=2)
                aux["spectral_gate"] = spectral_gate
            x = global_tokens + self.roi_residual_scale * (roi_fused - global_tokens)
            aux["roi_tokens"] = roi_tokens
            aux["roi_weights"] = roi_weights
            aux["roi_quality"] = quality_features
        else:
            x = torch.mean(x,4)
            x = torch.mean(x,3)
            x = rearrange(x, 'b c t -> b t c')

        if self.use_periodic:
            x, periodic_aux = self.periodic_modulator(x, token_fs)
            aux.update(periodic_aux)
        if self.return_aux:
            aux["token_fs"] = torch.full((B,), token_fs, dtype=x.dtype, device=x.device)

        for blk in self.blocks:
            x = blk(x)

        rPPG = x.permute(0,2,1) 
        rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG)    #[N, 1, D]
        rPPG = rPPG.squeeze(1)

        if self.return_aux:
            return rPPG, aux
        return rPPG
