from __future__ import annotations
import dataclasses
import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
from torch import nn


class BDHTokenizer:
    vocab_size: int = 256

    @classmethod
    def from_pretrained(cls, _name: str = "bdh-base") -> "BDHTokenizer":
        return cls()

    def encode(self, text: str) -> List[int]:
        b = (text or "").encode("utf-8", errors="ignore")
        return list(b) if b else [32]

    def decode(self, ids: List[int]) -> str:
        return bytes([int(x) & 255 for x in ids]).decode("utf-8", errors="ignore")


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def _get_freqs(n: int, theta: float, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))) / (2 * math.pi)


class _Attention(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(_get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))

    @staticmethod
    def _phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def _rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = _Attention._phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V):
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = (torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)) * self.freqs
        QR = self._rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = _Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, return_hidden: bool = False):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)  # (B,1,T,D)
        x = self.ln(x)

        for _ in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = self.drop(x_sparse * y_sparse)
            yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder)
            y = self.ln(yMLP)
            x = self.ln(x + y)

        hidden = x.view(B, T, D)
        logits = hidden @ self.lm_head

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_hidden:
            return logits, loss, hidden
        return logits, loss, None


@torch.no_grad()
def bdh_embed_text(
    model: BDH,
    tok: BDHTokenizer,
    text: str,
    device: torch.device,
    max_len: int = 512,
    windows: int = 3,
    pool: str = "mean",
    normalize: bool = True,
) -> torch.Tensor:
    ids = tok.encode(text)
    if len(ids) <= max_len:
        spans = [ids]
    else:
        if windows <= 1:
            spans = [ids[:max_len]]
        else:
            mid = max(0, (len(ids)//2) - (max_len//2))
            spans = [ids[:max_len], ids[mid:mid+max_len], ids[-max_len:]]

    embs = []
    for s in spans:
        x = torch.tensor(s, dtype=torch.long, device=device).unsqueeze(0)
        _, _, h = model(x, return_hidden=True)
        e = h[:, -1, :] if pool == "last" else h.mean(dim=1)
        embs.append(e[0])

    out = torch.stack(embs, dim=0).mean(dim=0)
    if normalize:
        out = F.normalize(out, dim=-1)
    return out


def save_ckpt(path: str, model: BDH, cfg: BDHConfig, extra: Dict | None = None):
    payload = {"config": cfg.__dict__, "state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ckpt(path: str, device: torch.device):
    obj = torch.load(path, map_location="cpu")
    cfg = BDHConfig(**obj["config"])
    m = BDH(cfg)
    m.load_state_dict(obj["state_dict"], strict=True)
    m.to(device).eval()
    extra = {k: v for k, v in obj.items() if k not in ("config", "state_dict")}
    return m, cfg, extra
