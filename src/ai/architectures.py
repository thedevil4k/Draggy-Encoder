"""Auto-build Real-ESRGAN and Real-CUGAN from state_dict key patterns.

Matches the exact key naming found in the official .pth checkpoints.

Torch is imported lazily so the module can be imported even when PyTorch
is not installed (e.g. Python 3.14 without wheels).  Only
``build_model_from_state_dict`` actually needs torch; everything else
is a no-op until that call.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import -- guard at module level so importing this module
# never crashes, even on Python versions where torch has no wheels.
# ---------------------------------------------------------------------------

try:
    # Nunca caer al torch del sistema: la aplicación solo reconoce el
    # runtime descargado en bin/runtimes/ai_modules.
    from src.runtimes.manager import import_local_runtime
    torch = import_local_runtime("torch")
    nn = torch.nn
    F = torch.nn.functional
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None      # type: ignore[assignment]
    nn = None         # type: ignore[assignment]
    F = None          # type: ignore[assignment]


def is_torch_available() -> bool:
    """Return True if PyTorch can be imported, False otherwise."""
    return _TORCH_AVAILABLE


def _require_torch():
    """Raise ImportError with a user-friendly message if torch is missing."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed.  Install it from Manage Dependencies → "
            "Runtimes & Binaries, or use Python 3.11–3.14 for automatic download."
        )


# ---------------------------------------------------------------------------
# Real-ESRGAN RRDBNet  (only defined when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _RDB(nn.Module):
        """Residual Dense Block used in Real-ESRGAN / ESRGAN.

        State dict keys (inside ``body.N``):
            rdb1.conv{1..5}.weight/bias
            rdb2.conv{1..5}.weight/bias
            rdb3.conv{1..5}.weight/bias
        """

        def __init__(self, num_feat: int = 64, num_growth: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_growth, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(num_feat + num_growth, num_growth, 3, 1, 1, bias=True)
            self.conv3 = nn.Conv2d(num_feat + num_growth * 2, num_growth, 3, 1, 1, bias=True)
            self.conv4 = nn.Conv2d(num_feat + num_growth * 3, num_growth, 3, 1, 1, bias=True)
            self.conv5 = nn.Conv2d(num_feat + num_growth * 4, num_feat, 3, 1, 1, bias=True)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
            return x5 * 0.2 + x

    class _RRDB(nn.Module):
        def __init__(self, num_feat: int = 64, num_growth: int = 32):
            super().__init__()
            self.rdb1 = _RDB(num_feat, num_growth)
            self.rdb2 = _RDB(num_feat, num_growth)
            self.rdb3 = _RDB(num_feat, num_growth)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            # OJO (bug real corregido): el original de Real-ESRGAN NO aplica
            # activación aquí, sólo escala el residuo:
            #     out = rdb3(rdb2(rdb1(x))); return out * 0.2 + x
            # Con un LeakyReLU extra después de la suma, cada uno de los 23
            # bloques iba apagando la señal negativa (x0.2) y la salida salía
            # oscura y sin contraste: medido con RealESRGAN_x4plus en este
            # equipo, la media caía de 221 a 126 y la energía de bordes a
            # 0.52x de la de un reescalado bilineal (es decir, MÁS borroso).
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        """Real-ESRGAN RRDBNet that uses the exact key names of the official checkpoints.

        Expected keys (all with ``.weight``/``.bias``):
            conv_first
            body.N.rdb{1,2,3}.conv{1,2,3,4,5}
            conv_body
            conv_up{1,2,3}
            conv_hr
            conv_last
        """

        def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3,
                     scale: int = 4, num_feat: int = 64,
                     num_growth: int = 32, num_block: int = 23,
                     num_feat_up: int = 0):
            super().__init__()
            self.scale = scale
            self.num_feat = num_feat
            self.num_feat_up = num_feat_up
            if num_feat_up <= 0:
                num_feat_up = num_feat * 4
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
            self.body = nn.ModuleList([_RRDB(num_feat, num_growth) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

            # Upsampling layers must be top-level attributes to match official
            # state_dict keys like ``conv_up1.weight``.
            num_up = int(math.log(scale, 2))
            for i in range(1, num_up + 1):
                setattr(self, f"conv_up{i}",
                        nn.Conv2d(num_feat, num_feat_up, 3, 1, 1, bias=True))

            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)

        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = feat
            for block in self.body:
                body_feat = block(body_feat)
            body_feat = self.conv_body(body_feat)
            feat = feat + body_feat

            # Upsampling layers. Two variants exist:
            #   - num_feat_up == num_feat * 4  ->  PixelShuffle(2) halves channels
            #   - num_feat_up == num_feat      ->  F.interpolate nearest (no channel change)
            #
            # ORDEN (importante, era un bug real): el original de Real-ESRGAN
            # hace SIEMPRE ``conv_up(interpolate(feat))`` — primero sube la
            # resolución y después convoluciona para refinar. Al revés
            # (convolucionar en baja resolución y luego interpolar con vecino)
            # la salida sale oscurecida y MÁS BORROSA que un reescalado
            # bilineal: medido con RealESRGAN_x4plus, la media caía de 221 a
            # 126 y la energía de bordes a 0.52x de la del bilineal.
            num_up = int(math.log(self.scale, 2))
            use_pixel_shuffle = (self.num_feat_up > self.num_feat)
            for i in range(1, num_up + 1):
                up_layer = getattr(self, f"conv_up{i}")
                if use_pixel_shuffle:
                    feat = self.lrelu(F.pixel_shuffle(up_layer(feat), 2))
                else:
                    feat = self.lrelu(up_layer(
                        F.interpolate(feat, scale_factor=2, mode="nearest")
                    ))

            feat = self.lrelu(self.conv_hr(feat))
            return self.conv_last(feat)


def _build_rrdbnet(sd: dict):
    """Build RRDBNet matching the checkpoint's state_dict keys exactly."""
    _require_torch()

    num_in_ch = sd["conv_first.weight"].shape[1]
    mid_ch = sd["conv_first.weight"].shape[0]

    # Count blocks: body.N.rdb1.conv1.weight
    block_ids = set()
    for k in sd:
        if k.startswith("body.") and ".rdb1.conv1.weight" in k:
            block_ids.add(int(k.split(".")[1]))
    n_blocks = max(block_ids) + 1 if block_ids else 23

    # Scale from number of conv_upN keys.
    up_keys = sorted(
        k for k in sd if k.startswith("conv_up") and k.endswith(".weight")
    )
    n_up = len(up_keys)
    scale = {1: 2, 2: 4, 3: 8}.get(n_up, 4)

    # Infer growth from rdb1.conv1.weight shape.
    growth = sd["body.0.rdb1.conv1.weight"].shape[0]

    # Detect conv_up output channels (varies by model variant).
    num_feat_up = mid_ch
    if up_keys:
        num_feat_up = sd[up_keys[0]].shape[0]

    logger.info(
        "RRDBNet: in_ch=%d, mid_ch=%d, blocks=%d, scale=%d, growth=%d, n_up=%d, feat_up=%d",
        num_in_ch, mid_ch, n_blocks, scale, growth, n_up, num_feat_up,
    )

    model = RRDBNet(
        num_in_ch=num_in_ch,
        num_out_ch=3,
        scale=scale,
        num_feat=mid_ch,
        num_growth=growth,
        num_block=n_blocks,
        num_feat_up=num_feat_up,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    logger.info(
        "RRDBNet loaded OK: %d params", sum(p.numel() for p in model.parameters())
    )
    return model


# ---------------------------------------------------------------------------
# SRVGGNetCompact — lightweight VGG-style SR (RealESR AnimeVideo v3, General)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class SRVGGNetCompact(nn.Module):
        """Compact VGG-style network for super-resolution.

        Used by RealESR AnimeVideo v3 and realesr-general-x4v3.
        Performs upsampling in the last layer via PixelShuffle.
        Learns the residual (adds nearest-upsampled input to output).

        State dict keys (flat sequence):
            body.0.weight  — first Conv2d(in, feat)
            body.0.bias
            body.1.weight  — PReLU (num_parameters=feat)
            body.2.weight  — Conv2d(feat, feat)
            body.2.bias
            ...
            body.{2*num_conv+1}.weight — last Conv2d(feat, out*scale^2)
            body.{2*num_conv+1}.bias
        """

        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_conv=16, upscale=4, act_type='prelu'):
            super().__init__()
            self.num_in_ch = num_in_ch
            self.num_out_ch = num_out_ch
            self.num_feat = num_feat
            self.num_conv = num_conv
            self.upscale = upscale
            self.act_type = act_type

            self.body = nn.ModuleList()
            # first conv
            self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            # first activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            else:
                activation = nn.ReLU(inplace=True)
            self.body.append(activation)

            # body convs
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                if act_type == 'relu':
                    activation = nn.ReLU(inplace=True)
                elif act_type == 'prelu':
                    activation = nn.PReLU(num_parameters=num_feat)
                elif act_type == 'leakyrelu':
                    activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                else:
                    activation = nn.ReLU(inplace=True)
                self.body.append(activation)

            # last conv (outputs out_ch * upscale^2 channels)
            self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            # residual: add nearest-upsampled input
            base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
            out += base
            return out


def _build_srvgg(sd: dict):
    """Build SRVGGNetCompact from a state_dict."""
    _require_torch()

    # Detect parameters from state_dict shapes
    num_in_ch = sd["body.0.weight"].shape[1]
    num_feat = sd["body.0.weight"].shape[0]

    # Count conv layers: body.0 is first conv, body.{last} is last conv.
    # PReLU keys have shape (num_feat,), conv keys have shape (out, in, 3, 3).
    # last key is body.{2*num_conv+1}
    last_key = max(
        (k for k in sd if k.startswith("body.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    last_idx = int(last_key.split(".")[1])
    num_conv = (last_idx - 1) // 2  # first conv at 0, then pairs of (conv, act)

    # Output channels from last conv
    num_out_ch_times_scale2 = sd[last_key].shape[0]
    # Detect scale from the relationship: out_ch * scale^2
    # Try larger scales first to avoid false matches (e.g. 48 % 12 == 0 for scale=2)
    scale = 4
    for s in (8, 4, 3, 2):
        if num_out_ch_times_scale2 % (3 * s * s) == 0:
            scale = s
            break
    num_out_ch = num_out_ch_times_scale2 // (scale * scale)

    # Detect activation type
    act_type = 'prelu'
    prelu_key = f"body.1.weight"
    if prelu_key in sd and sd[prelu_key].ndim == 1:
        act_type = 'prelu'

    logger.info(
        "SRVGGNetCompact: in_ch=%d, out_ch=%d, feat=%d, conv=%d, scale=%d, act=%s",
        num_in_ch, num_out_ch, num_feat, num_conv, scale, act_type,
    )

    model = SRVGGNetCompact(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=scale,
        act_type=act_type,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    logger.info(
        "SRVGGNetCompact loaded OK: %d params",
        sum(p.numel() for p in model.parameters()),
    )
    return model


# ---------------------------------------------------------------------------
# OmniSR — lightweight ViT-based SR (CVPR 2023)
# Pure PyTorch implementation (no einops dependency).
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _LayerNorm2d(nn.Module):
        """Channel-wise LayerNorm for 4D tensors."""

        def __init__(self, num_channels, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
            self.eps = eps

        def forward(self, x):
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            return x

    class _OmniESA(nn.Module):
        """Enhanced Spatial Attention used by OmniSR."""

        def __init__(self, esa_channels, n_feats):
            super().__init__()
            f = esa_channels
            self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
            self.conv_f = nn.Conv2d(f, f, kernel_size=1)
            self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
            self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            c1_ = self.conv1(x)
            c1 = self.conv2(c1_)
            v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
            c3 = self.conv3(v_max)
            c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            cf = self.conv_f(c1_)
            c4 = self.conv4(c3 + cf)
            m = self.sigmoid(c4)
            return x * m

    class _PixelShuffleBlock(nn.Module):
        """Conv + PixelShuffle upsampling block."""

        def __init__(self, in_channels, out_channels, upscale_factor):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2),
                                  kernel_size=3, padding=1, bias=False)
            self.ps = nn.PixelShuffle(upscale_factor)

        def forward(self, x):
            return self.ps(self.conv(x))

    class _MBConv(nn.Module):
        """Mobile Bottleneck Convolution block.

        Checkpoint keys: layer.N.fn.{0,2,4.gate.{1,3},5}.weight/bias
        Structure:
          fn.0  = Conv2d(dim, hidden, 1)           project_in
          fn.1  = GELU
          fn.2  = Conv2d(hidden, hidden, 3, groups=hidden)  dwconv
          fn.3  = GELU
          fn.4  = SE block:
            fn.4.gate.0 = AdaptiveAvgPool2d(1)
            fn.4.gate.1 = Linear(hidden, se_dim)
            fn.4.gate.2 = GELU
            fn.4.gate.3 = Linear(se_dim, hidden)
          fn.5  = Conv2d(hidden, dim, 1)           project_out
        """

        def __init__(self, dim, expansion_rate=1, shrinkage_rate=0.25):
            super().__init__()
            hidden_dim = int(expansion_rate * dim)
            se_dim = int(hidden_dim * shrinkage_rate)
            self.fn = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, 1),           # fn.0
                nn.GELU(),                                # fn.1
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),  # fn.2
                nn.GELU(),                                # fn.3
            )
            # SE block: gate Sequential with Pool+Linear+GELU+Linear
            se_gate = nn.Sequential()
            se_gate.add_module("0", nn.AdaptiveAvgPool2d(1))   # gate.0
            se_gate.add_module("1", nn.Linear(hidden_dim, se_dim, bias=False))  # gate.1
            se_gate.add_module("2", nn.GELU())                  # gate.2
            se_gate.add_module("3", nn.Linear(se_dim, hidden_dim, bias=False))  # gate.3
            se = nn.Sequential()
            se.add_module("gate", se_gate)
            self.fn.add_module("4", se)
            self.fn.add_module("5", nn.Conv2d(hidden_dim, dim, 1))  # fn.5
            self.dim = dim
            self.hidden_dim = hidden_dim

        def forward(self, x):
            residual = x
            # Manual forward to match checkpoint SE structure exactly
            x = self.fn[0](x)   # conv project_in
            x = self.fn[1](x)   # gelu
            x = self.fn[2](x)   # dwconv
            x = self.fn[3](x)   # gelu
            # SE block
            se = self.fn[4].gate[0](x)  # AdaptiveAvgPool2d -> (B, hidden, 1, 1)
            se = se.flatten(1)           # Flatten -> (B, hidden)
            se = self.fn[4].gate[1](se)  # Linear(hidden, se_dim)
            se = self.fn[4].gate[2](se)  # GELU
            se = self.fn[4].gate[3](se)  # Linear(se_dim, hidden)
            se = se.unsqueeze(-1).unsqueeze(-1)  # (B, hidden, 1, 1)
            x = x * se
            x = self.fn[5](x)   # conv project_out
            return x + residual

    class _ChannelAttention(nn.Module):
        """Channel self-attention with window partitioning.

        Checkpoint keys: layer.N.{norm.{weight,bias}, fn.{temperature, qkv, qkv_dwconv, project_out}}
        """

        def __init__(self, dim, heads=4, window_size=8):
            super().__init__()
            self.heads = heads
            self.ps = window_size
            self.norm = _LayerNorm2d(dim)
            self.fn = nn.Sequential()
            self.fn.temperature = nn.Parameter(torch.ones(heads, 1, 1))
            self.fn.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
            self.fn.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                            stride=1, padding=1, groups=dim * 3, bias=False)
            self.fn.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        def forward(self, x):
            x = self.norm(x)
            b, c, h, w = x.shape
            ps = self.ps
            qkv = self.fn.qkv_dwconv(self.fn.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = q.view(b, self.heads, c // self.heads, h // ps, ps, w // ps, ps)
            q = q.permute(0, 3, 5, 1, 4, 6, 2).contiguous().view(
                b * (h // ps) * (w // ps), self.heads, ps * ps, c // self.heads)
            k = k.view(b, self.heads, c // self.heads, h // ps, ps, w // ps, ps)
            k = k.permute(0, 3, 5, 1, 4, 6, 2).contiguous().view(
                b * (h // ps) * (w // ps), self.heads, ps * ps, c // self.heads)
            v = v.view(b, self.heads, c // self.heads, h // ps, ps, w // ps, ps)
            v = v.permute(0, 3, 5, 1, 4, 6, 2).contiguous().view(
                b * (h // ps) * (w // ps), self.heads, ps * ps, c // self.heads)
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * self.fn.temperature
            attn = attn.softmax(dim=-1)
            out = attn @ v
            out = out.view(b, h // ps, w // ps, self.heads, ps * ps, c // self.heads)
            out = out.permute(0, 3, 5, 1, 4, 2).contiguous().view(b, c, h, w)
            return self.fn.project_out(out)

    class _ChannelAttentionGrid(nn.Module):
        """Grid-like channel attention (global interactions).

        Checkpoint keys: layer.N.{norm.{weight,bias}, fn.{project_in, dwconv, project_out}}
        Note: In the actual checkpoint, ChannelAttentionGrid has the same
        parameter structure as GatedConvFFN (project_in, dwconv, project_out).
        """

        def __init__(self, dim, heads=4, window_size=8):
            super().__init__()
            self.norm = _LayerNorm2d(dim)
            self.fn = nn.Sequential()
            self.fn.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
            self.fn.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                        stride=1, padding=1, groups=dim * 2, bias=False)
            self.fn.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        def forward(self, x):
            x = self.norm(x)
            x = self.fn.project_in(x)
            x1, x2 = self.fn.dwconv(x).chunk(2, dim=1)
            return self.fn.project_out(F.gelu(x1) * x2)

    class _WindowAttention(nn.Module):
        """Window-based spatial self-attention with optional positional encoding.

        Checkpoint keys: layer.N.{norm.{weight,bias}, fn.{to_qkv, to_out.0, rel_pos_bias}}
        Note: to_qkv and to_out use nn.Linear (2D weight tensors), not Conv2d.
        """

        def __init__(self, dim, dim_head=32, window_size=8, with_pe=True):
            super().__init__()
            self.heads = dim // dim_head
            self.scale = dim_head ** -0.5
            self.with_pe = with_pe
            self.ps = window_size
            self.norm = _LayerNorm2d(dim)
            self.fn = nn.Sequential()
            self.fn.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.fn.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False))
            if self.with_pe:
                self.fn.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
                pos = torch.arange(window_size)
                grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
                grid_flat = grid.reshape(2, -1).t()
                rel_pos = grid_flat.unsqueeze(1) - grid_flat.unsqueeze(0)
                rel_pos += window_size - 1
                indices = rel_pos[..., 0] * (2 * window_size - 1) + rel_pos[..., 1]
                self.register_buffer('rel_pos_indices', indices, persistent=False)

        def forward(self, x):
            x = self.norm(x)
            b, c, h, w = x.shape
            ps = self.ps
            nh = self.heads
            hd = c // nh
            # Linear expects (B, N, C) so flatten spatial dims
            x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            qkv = self.fn.to_qkv(x_flat)  # (B, H*W, 3*C)
            q, k, v = qkv.chunk(3, dim=-1)
            # Reshape to (B, num_windows_h, num_windows_w, heads, ps*ps, hd)
            q = q.view(b, h // ps, ps, w // ps, ps, nh, hd)
            q = q.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            q = q.view(b * (h // ps) * (w // ps), nh, ps * ps, hd)
            k = k.view(b, h // ps, ps, w // ps, ps, nh, hd)
            k = k.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            k = k.view(b * (h // ps) * (w // ps), nh, ps * ps, hd)
            v = v.view(b, h // ps, ps, w // ps, ps, nh, hd)
            v = v.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            v = v.view(b * (h // ps) * (w // ps), nh, ps * ps, hd)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.with_pe:
                bias = self.fn.rel_pos_bias(self.rel_pos_indices)
                attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
            attn = F.softmax(attn, dim=-1)
            out = attn @ v
            out = out.view(b, h // ps, w // ps, nh, ps, ps, hd)
            out = out.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            out = out.view(b, c, h, w)
            # Linear expects (B, N, C), flatten spatial dims
            out_flat = out.flatten(2).transpose(1, 2)  # (B, H*W, C)
            out_flat = self.fn.to_out(out_flat)  # (B, H*W, C)
            out = out_flat.transpose(1, 2).view(b, c, h, w)
            return out

    class _GatedConvFFN(nn.Module):
        """Gated Convolutional Feed-Forward Network.

        Checkpoint keys: layer.N.{norm.{weight,bias}, fn.{project_in, dwconv, project_out}}
        """

        def __init__(self, dim):
            super().__init__()
            self.norm = _LayerNorm2d(dim)
            self.fn = nn.Sequential()
            self.fn.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
            self.fn.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                        stride=1, padding=1, groups=dim * 2, bias=False)
            self.fn.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        def forward(self, x):
            x = self.norm(x)
            x = self.fn.project_in(x)
            x1, x2 = self.fn.dwconv(x).chunk(2, dim=1)
            return self.fn.project_out(F.gelu(x1) * x2)

    class _OSABlock(nn.Module):
        """Omni Self-Attention Block.

        13-layer structure matching the official OmniSR checkpoint keys:
        0: MBConv, 1: Identity, 2: WindowAttention, 3: Identity,
        4: GatedConvFFN, 5: ChannelAttention, 6: GatedConvFFN,
        7: Identity, 8: WindowAttention, 9: Identity,
        10: ChannelAttentionGrid, 11: ChannelAttention, 12: ChannelAttentionGrid
        """

        def __init__(self, channel_num, window_size=8, with_pe=False):
            super().__init__()
            self.layer = nn.Sequential(
                _MBConv(channel_num, expansion_rate=1, shrinkage_rate=0.25),
                nn.Identity(),
                _WindowAttention(channel_num, dim_head=channel_num // 4,
                                 window_size=window_size, with_pe=with_pe),
                nn.Identity(),
                _GatedConvFFN(channel_num),
                _ChannelAttention(channel_num, heads=4, window_size=window_size),
                _GatedConvFFN(channel_num),
                nn.Identity(),
                _WindowAttention(channel_num, dim_head=channel_num // 4,
                                 window_size=window_size, with_pe=with_pe),
                nn.Identity(),
                _ChannelAttentionGrid(channel_num, heads=4, window_size=window_size),
                _ChannelAttention(channel_num, heads=4, window_size=window_size),
                _ChannelAttentionGrid(channel_num, heads=4, window_size=window_size),
            )

        def forward(self, x):
            return self.layer(x)

    class _OSAG(nn.Module):
        """Omni-Scale Aggregation Group."""

        def __init__(self, channel_num=64, block_num=4, window_size=8, with_pe=False):
            super().__init__()
            blocks = []
            for _ in range(block_num):
                blocks.append(_OSABlock(channel_num, window_size=window_size, with_pe=with_pe))
            blocks.append(nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=True))
            self.residual_layer = nn.Sequential(*blocks)
            esa_ch = max(channel_num // 4, 16)
            self.esa = _OmniESA(esa_ch, channel_num)

        def forward(self, x):
            out = self.residual_layer(x)
            out = out + x
            return self.esa(out)

    class OmniSR(nn.Module):
        """Omni Aggregation Networks for Lightweight Image Super-Resolution.

        CVPR 2023. Architecture from Francis0625/Omni-SR.
        Pure PyTorch implementation (no einops dependency).
        State dict keys: input.*, output.*, up.*, residual_layer.*.residual_layer.*
        """

        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                     res_num=5, block_num=4, up_scale=4, bias=True,
                     window_size=8, with_pe=True):
            super().__init__()
            self.input = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=bias)
            self.output = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias)
            # Use nn.Sequential to match checkpoint keys: up.0.weight, up.0.bias
            self.up = nn.Sequential(
                nn.Conv2d(num_feat, num_out_ch * (up_scale ** 2), 3, 1, 1, bias=bias),
                nn.PixelShuffle(up_scale),
            )
            self.residual_layer = nn.ModuleList([
                _OSAG(channel_num=num_feat, block_num=block_num,
                      window_size=window_size, with_pe=with_pe)
                for _ in range(res_num)
            ])
            self.window_size = window_size
            self.up_scale = up_scale

        def check_image_size(self, x):
            _, _, h, w = x.size()
            mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
            mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
            if mod_pad_h or mod_pad_w:
                x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
            return x

        def forward(self, x):
            H, W = x.shape[2:]
            x = self.check_image_size(x)
            residual = self.input(x)
            out = residual
            for layer in self.residual_layer:
                out = layer(out)
            out = self.output(out) + residual
            out = self.up(out)
            return out[:, :, :H * self.up_scale, :W * self.up_scale]


def _build_omnisr(sd: dict):
    """Build OmniSR from a state_dict."""
    _require_torch()

    # Filter out profiling metadata keys (total_ops, total_params)
    sd = {k: v for k, v in sd.items() if "total_ops" not in k and "total_params" not in k}

    num_in_ch = sd["input.weight"].shape[1]
    num_feat = sd["input.weight"].shape[0]

    if "up.0.weight" in sd:
        out_per_scale = sd["up.0.weight"].shape[0]
        num_out_ch = 3
        scale_sq = out_per_scale // num_out_ch
        scale = int(round(scale_sq ** 0.5))
    else:
        scale = 4
        num_out_ch = 3

    # Count OSAG blocks (residual_layer.0 through residual_layer.N)
    res_num = 0
    for k in sd:
        if k.startswith("residual_layer."):
            idx = k.split(".")[1]
            if idx.isdigit():
                res_num = max(res_num, int(idx) + 1)

    window_size = 8

    logger.info(
        "OmniSR: in_ch=%d, feat=%d, scale=%d, res_num=%d",
        num_in_ch, num_feat, scale, res_num,
    )

    model = OmniSR(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        res_num=res_num,
        block_num=1,
        up_scale=scale,
        bias=True,
        window_size=window_size,
        with_pe=True,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    logger.info(
        "OmniSR loaded OK: %d params",
        sum(p.numel() for p in model.parameters()),
    )
    return model


# ---------------------------------------------------------------------------
# Real-CUGAN CompactUNet  (only defined when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _ConvB(nn.Module):
        """Conv -> LeakyReLU -> Conv (matches conv.0, conv.2 naming)."""

        def __init__(self, in_ch, mid_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=True),
            )

        def forward(self, x):
            return self.conv(x)

    class _SE(nn.Module):
        """Squeeze-and-Excitation (matches seblock.conv1, seblock.conv2 naming)."""

        def __init__(self, ch, mid):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, mid, 1, bias=True)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(mid, ch, 1, bias=True)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            return x * self.sig(self.conv2(self.relu(self.conv1(x))))


def _build_cugan(sd: dict):
    """Build CUGAN matching the checkpoint's state_dict keys exactly."""
    _require_torch()

    def _out(key):
        return sd[key].shape[0]

    def _in(key):
        return sd[key].shape[1]

    class UNet1(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = _ConvB(3, _out("unet1.conv1.conv.0.weight"), _out("unet1.conv1.conv.2.weight"))
            self.conv1_down = nn.Conv2d(_out("unet1.conv1_down.weight"), _out("unet1.conv1_down.weight"), 2, 2)
            self.conv2 = _ConvB(_out("unet1.conv1_down.weight"), _out("unet1.conv2.conv.0.weight"), _out("unet1.conv2.conv.2.weight"))
            self.conv2.seblock = _SE(_out("unet1.conv2.seblock.conv2.weight"), _out("unet1.conv2.seblock.conv1.weight"))
            self.conv2_up = nn.ConvTranspose2d(_out("unet1.conv2_up.weight"), _out("unet1.conv2_up.weight"), 2, 2)
            self.conv3 = nn.Conv2d(_in("unet1.conv3.weight"), _out("unet1.conv3.weight"), 3, 1, 1, bias=True)
            self.conv_bottom = nn.ConvTranspose2d(_out("unet1.conv_bottom.weight"), _out("unet1.conv_bottom.weight"), 4, 2, 1)

        def forward(self, x):
            e1 = self.conv1(x)
            e2 = self.conv2(self.conv1_down(e1))
            e2 = self.conv2.seblock(e2)
            d = self.conv3(e1 + self.conv2_up(e2))
            return self.conv_bottom(d)

    class UNet2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = _ConvB(_out("unet1.conv_bottom.weight"), _out("unet2.conv1.conv.0.weight"), _out("unet2.conv1.conv.2.weight"))
            self.conv1_down = nn.Conv2d(_out("unet2.conv1_down.weight"), _out("unet2.conv1_down.weight"), 2, 2)
            self.conv2 = _ConvB(_out("unet2.conv1_down.weight"), _out("unet2.conv2.conv.0.weight"), _out("unet2.conv2.conv.2.weight"))
            self.conv2.seblock = _SE(_out("unet2.conv2.seblock.conv2.weight"), _out("unet2.conv2.seblock.conv1.weight"))
            self.conv2_down = nn.Conv2d(_out("unet2.conv2_down.weight"), _out("unet2.conv2_down.weight"), 2, 2)
            self.conv3 = _ConvB(_out("unet2.conv2_down.weight"), _out("unet2.conv3.conv.0.weight"), _out("unet2.conv3.conv.2.weight"))
            self.conv3.seblock = _SE(_out("unet2.conv3.seblock.conv2.weight"), _out("unet2.conv3.seblock.conv1.weight"))
            self.conv3_up = nn.ConvTranspose2d(_out("unet2.conv3_up.weight"), _out("unet2.conv3_up.weight"), 2, 2)
            self.conv4 = _ConvB(_out("unet2.conv3_up.weight"), _out("unet2.conv4.conv.0.weight"), _out("unet2.conv4.conv.2.weight"))
            self.conv4.seblock = _SE(_out("unet2.conv4.seblock.conv2.weight"), _out("unet2.conv4.seblock.conv1.weight"))
            self.conv4_up = nn.ConvTranspose2d(_out("unet2.conv4_up.weight"), _out("unet2.conv4_up.weight"), 2, 2)
            self.conv5 = nn.Conv2d(_out("unet2.conv4_up.weight"), _out("unet2.conv5.weight"), 3, 1, 1, bias=True)
            self.conv_bottom = nn.Conv2d(_out("unet2.conv5.weight"), _out("unet2.conv_bottom.weight"), 3, 1, 1, bias=True)

        def forward(self, x):
            e1 = self.conv1(x)
            e2 = self.conv2(self.conv1_down(e1))
            e2 = self.conv2.seblock(e2)
            e3 = self.conv3(self.conv2_down(e2))
            e3 = self.conv3.seblock(e3)
            d2 = self.conv4(e2 + self.conv3_up(e3))
            d2 = self.conv4.seblock(d2)
            d1 = self.conv5(e1 + self.conv4_up(d2))
            return self.conv_bottom(d1)

    class CUGAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet1 = UNet1()
            self.unet2 = UNet2()
            out_ch = _out("conv_final.weight")
            in_ch = _out("unet2.conv_bottom.weight")
            self.conv_final = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)

        def forward(self, x):
            f1 = self.unet1(x)
            f2 = self.unet2(f1)
            return F.pixel_shuffle(self.conv_final(f2), 2)

    model = CUGAN()
    model.load_state_dict(sd, strict=True)
    model.eval()
    scale = int(math.sqrt(_out("conv_final.weight") / 3))
    logger.info(
        "CUGAN loaded OK: %d params, scale=%d",
        sum(p.numel() for p in model.parameters()), scale,
    )
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Estas tres viven en ``arch_specs`` (sin torch) porque la UI las necesita a
# nivel de módulo y este módulo SÍ importa torch: se re-exportan aquí para
# mantener la API histórica (tests y llamadores existentes).
from src.ai.arch_specs import (  # noqa: E402  (re-export deliberado)
    ONNX_CONVERTIBLE_ARCHS,
    arch_is_onnx_convertible,
    onnx_convertible_archs,
)


def detect_model_arch_and_scale(state_dict: dict):
    """Detecta ``(arch, scale)`` de un state_dict SIN construir el modelo.

    Espeja la lógica de ``build_model_from_state_dict`` pero solo inspecciona
    claves/formas de tensores, así que no necesita torch ni GPU (un dict
    cargado con ``torch.load(..., weights_only=True)`` o safetensors sirve).
    Devuelve ``None`` si no reconoce la arquitectura.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return None

    # Unwrap de checkpoints Real-ESRGAN envueltos (params_ema/params/state_dict)
    if "body.0.rdb1.conv1.weight" not in state_dict:
        for wrapper_key in ("params_ema", "params", "state_dict"):
            if wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
                state_dict = state_dict[wrapper_key]
                break

    keys = list(state_dict.keys())

    def _shape(key):
        return getattr(state_dict[key], "shape", None)

    # OmniSR: input.weight / output.weight / residual_layer.*
    if ("input.weight" in keys and "output.weight" in keys
            and any(k.startswith("residual_layer.") for k in keys)):
        try:
            out_per_scale = _shape("up.0.weight")[0]
            scale = int(round((out_per_scale // 3) ** 0.5))
        except Exception:
            scale = 4
        return ("omnisr", scale)

    if any(k.startswith("body.") for k in keys):
        if any("rdb" in k for k in keys):
            # RRDBNet: scale desde número de conv_upN
            ups = [k for k in keys if k.startswith("conv_up") and k.endswith(".weight")]
            scale = {1: 2, 2: 4, 3: 8}.get(len(ups), 4)
            return ("rrdb", scale)
        # SRVGGNetCompact: última body.N.weight -> out_ch * scale^2
        try:
            last_key = max(
                (k for k in keys if k.startswith("body.") and k.endswith(".weight")),
                key=lambda k: int(k.split(".")[1]),
            )
            out_c = _shape(last_key)[0]
            scale = 4
            for s in (8, 4, 3, 2):
                if out_c % (3 * s * s) == 0:
                    scale = s
                    break
        except Exception:
            scale = 4
        return ("srvgg", scale)

    if any(k.startswith("unet") for k in keys):
        try:
            scale = int(round((_shape("conv_final.weight")[0] / 3) ** 0.5))
        except Exception:
            scale = 2
        return ("cugan", scale)

    return None


def detect_arch_and_scale(weight_path: str):
    """Lee un archivo de pesos (.pth/.safetensors) y devuelve
    ``(arch, scale)`` o ``(None, None)`` si no se puede detectar.

    Uso: tras descargar un motor de usuario del Hub, para rellenar el
    registro con arch/scale reales (el catálogo los declara estáticos).
    """
    import os
    if not weight_path or not os.path.isfile(weight_path):
        return None, None
    try:
        lower = weight_path.lower()
        if lower.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(weight_path)
        else:
            from src.runtimes.manager import import_local_runtime
            _torch = import_local_runtime("torch")
            sd = _torch.load(weight_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.info("detect_arch_and_scale: could not load %s: %s", weight_path, exc)
        return None, None
    detected = detect_model_arch_and_scale(sd)
    if not detected:
        logger.info("detect_arch_and_scale: unknown arch in %s", weight_path)
        return None, None
    return detected


# -----------------------------------------------------------------------
# Escala NATIVA de un fichero de pesos (la verdad, no la del nombre)
# -----------------------------------------------------------------------
#
# Un modelo no "admite" varias escalas: cada fichero hornea UNA en sus
# pesos. Por eso la UI no puede ofrecer x2/x3/x4 para un solo .onnx x4 --
# elegir x2 ahí no hace que la red produzca x2, sólo que se tire el
# resultado x4 hacia abajo (peor calidad y el doble de cómputo, ver
# perf-audit.md §3.5). Las varias escalas de una familia son VARIOS
# ficheros (OmniSR_X3/X4, RealESRGAN_x2plus/x4plus...), y cada uno con
# la suya. Estas funciones leen la escala del fichero de verdad.

# ``parse_scale_from_name`` vive en ``arch_specs`` (sin torch): la UI la
# necesita para listar modelos y este módulo importa torch al cargarse.
from src.ai.arch_specs import (  # noqa: E402  (re-export deliberado)
    parse_scale_from_name,
)


def _onnx_dim(value) -> int:
    """``dim_value`` de un eje ONNX, o 0 si es dinámico (``dim_param``)."""
    try:
        dim = int(value)
    except (TypeError, ValueError):
        return 0
    return dim if dim > 0 else 0


def _onnx_upsample_factors(onnx, graph) -> float:
    """Producto de los factores espaciales de subida de un grafo ONNX.

    Recorre los ops que cambian la resolución: ``Resize``/``Upsample`` (su
    eje H, índice 2 de ``scales``) y ``ConvTranspose`` (su ``strides``).
    ``DepthToSpace`` (PixelShuffle) NO cuenta: cambia canales por píxeles
    sin mover H/W, que es justo por lo que la red sube x4 en features y
    luego las reparte en un x4 de salida.

    Validado con los ONNX instalados: AnimeVideo v3 → 4 (Resize 4),
    General x4v3 → 4, OmniSR_X2_DIV2K → 8, OmniSR_X4_DIV2K → 16; los
    cuatro coinciden con un forward real.
    """
    from onnx import numpy_helper

    initializers = {}
    try:
        initializers = {
            i.name: numpy_helper.to_array(i) for i in graph.initializer
        }
    except Exception:
        pass
    # Constantes producidas por un nodo Constant (así exporta el conversor
    # de Real-ESRGAN: ``/Constant_output_0`` con scales=[1,1,4,4]).
    constants = {}
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    try:
                        constants[node.output[0]] = numpy_helper.to_array(attr.t)
                    except Exception:
                        pass

    def _scales_value(name):
        if not name:
            return None
        if name in initializers:
            return initializers[name]
        return constants.get(name)

    factor = 1.0
    for node in graph.node:
        if node.op_type in ("Resize", "Upsample"):
            scales = None
            for attr in node.attribute:
                if attr.name == "scales":
                    scales = onnx.helper.get_attribute_value(attr)
            if scales is None and len(node.input) > 2:
                scales = _scales_value(node.input[2])
            if scales is not None:
                try:
                    factor *= float(scales[2])
                except (TypeError, ValueError, IndexError):
                    continue
        elif node.op_type == "ConvTranspose":
            for attr in node.attribute:
                if attr.name == "strides":
                    try:
                        factor *= float(onnx.helper.get_attribute_value(attr)[2])
                    except (TypeError, ValueError, IndexError):
                        continue
    return factor


def detect_onnx_scale(onnx_path: str) -> int:
    """Escala nativa de un ``.onnx`` leyéndolo por dentro (sin inferencia).

    Primero mira las formas declaradas (si entrada y salida tienen H/W
    fijos, el cociente ES la escala). Si son dinámicas -- que es lo
    habitual en los SR exportados, p.ej. ``['batch',3,'height','width']``
    en ambos lados -- calcula el producto de los ops de subida del grafo.
    Devuelve 1 si no se puede determinar.
    """
    import os
    if not onnx_path or not os.path.isfile(onnx_path):
        return 1
    try:
        from src.runtimes.manager import import_local_runtime
        onnx = import_local_runtime("onnx")
    except Exception as exc:
        logger.info("detect_onnx_scale: onnx runtime unavailable (%s)", exc)
        return 1
    try:
        # load_external_data=False: sólo queremos la topología del grafo,
        # no los pesos (así se lee un modelo de 70 MB en milisegundos).
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        logger.info("detect_onnx_scale: could not read %s: %s", onnx_path, exc)
        return 1

    try:
        graph = model.graph
        in_dims = [_onnx_dim(d.dim_value) for d in graph.input[0].type.tensor_type.shape.dim]
        out_dims = [_onnx_dim(d.dim_value) for d in graph.output[0].type.tensor_type.shape.dim]
    except Exception:
        return 1

    # Formas fijas: el cociente de salida/entrada es la escala exacta.
    if len(in_dims) >= 4 and len(out_dims) >= 4 and in_dims[2] and out_dims[2]:
        scale = round(out_dims[2] / in_dims[2])
        if 1 < scale <= 64:
            return int(scale)

    try:
        scale = round(_onnx_upsample_factors(onnx, graph))
    except Exception as exc:
        logger.info("detect_onnx_scale: no graph scale in %s (%s)", onnx_path, exc)
        return 1
    return int(scale) if 1 < scale <= 64 else 1


def detect_model_file_scale(model_path: str) -> int:
    """Escala NATIVA de un fichero de pesos, sea del tipo que sea.

    ``.onnx`` → se lee el grafo. ``.pth/.pt/.bin/.safetensors`` → se leen
    las formas de los tensores. Devuelve 1 si no se puede determinar.
    """
    import os
    if not model_path or not os.path.isfile(model_path):
        return 1
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".onnx":
        return detect_onnx_scale(model_path)
    _arch, scale = detect_arch_and_scale(model_path)
    try:
        scale = int(scale)
    except (TypeError, ValueError):
        return 1
    return scale if 1 < scale <= 64 else 1


#: Sesión donde se cachea lo detectado (fingerprints, no pesos).
SCALES_FILENAME = "model_scales.json"

#: Sube esto si cambia la forma de detectar la escala.
SCALES_VERSION = 1

_SCALE_CACHE: dict = {}


def _scales_path() -> str:
    import os
    from src.config import ensure_runtime_dirs, get_session_dir
    ensure_runtime_dirs()
    return os.path.join(get_session_dir(), SCALES_FILENAME)


def _load_scale_cache() -> dict:
    import json
    try:
        with open(_scales_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_scale_cache(data: dict) -> None:
    import json
    import os
    try:
        path = _scales_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("could not persist model scales: %s", exc)


def model_file_fingerprint(model_path: str) -> str:
    """``tamaño_mtime`` del fichero: cambia si se re-descarga el modelo."""
    import os
    try:
        st = os.stat(model_path)
        return f"{st.st_size}_{st.st_mtime_ns}"
    except OSError:
        return ""


def get_model_file_scale(model_path: str, fallback: int = 0) -> int:
    """Escala nativa de un fichero con caché por huella (rápido en la UI).

    Es la función que usa el nodo Upscale al listar modelos: escanea la
    carpeta de un motor en cada refresco y no puede pagar 0.1 s por peso.
    La clave de caché es ``ruta|tamaño_mtime``, así que si el usuario
    sustituye un modelo por otro se vuelve a medir. Si el fichero no se
    puede inspeccionar devuelve ``fallback`` (el nombre, si el llamante
    lo deduce) y, en último término, 1.
    """
    import os
    if not model_path or not os.path.isfile(model_path):
        return max(1, int(fallback or 1))
    key = f"{os.path.normcase(os.path.abspath(model_path))}|{model_file_fingerprint(model_path)}"
    if key in _SCALE_CACHE:
        scale = _SCALE_CACHE[key]
    else:
        data = _load_scale_cache()
        if key in data:
            scale = int(data[key])
        else:
            scale = detect_model_file_scale(model_path)
            if scale > 1:
                data[key] = scale
                _save_scale_cache(data)
        _SCALE_CACHE[key] = scale
    if scale > 1:
        return scale
    return max(1, int(fallback or 1))


def build_model_from_state_dict(state_dict: dict, scale: int = 4):
    """Detect architecture from state_dict keys and build+load it.

    Requires PyTorch to be installed.  Raises ImportError with a clear
    message if torch is not available.
    """
    _require_torch()

    # Many official Real-ESRGAN checkpoints wrap the actual model weights
    # under a ``params_ema`` or ``params`` key.  Unwrap it transparently.
    if isinstance(state_dict, dict) and "body.0.rdb1.conv1.weight" not in state_dict:
        for wrapper_key in ("params_ema", "params", "state_dict"):
            if wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
                state_dict = state_dict[wrapper_key]
                break

    keys = list(state_dict.keys())
    # OmniSR: keys like input.weight, output.weight, residual_layer.*
    if "input.weight" in keys and "output.weight" in keys and any(k.startswith("residual_layer.") for k in keys):
        return _build_omnisr(state_dict)
    if any(k.startswith("body.") for k in keys):
        # Distinguish SRVGGNetCompact from RRDBNet:
        # SRVGGNetCompact: flat keys like body.0.weight, body.2.weight ...
        # RRDBNet: nested keys like body.0.rdb1.conv1.weight ...
        if any("rdb" in k for k in keys):
            return _build_rrdbnet(state_dict)
        else:
            return _build_srvgg(state_dict)
    if any(k.startswith("unet") for k in keys):
        return _build_cugan(state_dict)
    raise ValueError(f"Unknown model architecture. Keys: {keys[:5]}")
