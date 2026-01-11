import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses

@dataclasses.dataclass
class RealBDHConfig:
    n_neurons: int = 2048
    sparsity: float = 0.02
    vocab_size: int = 50257
    hidden_dim: int = 256
    num_labels: int = 2


class RealBDH(nn.Module):
    def __init__(self, config: RealBDHConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.n_neurons)

        # Sparse adjacency
        adj = (torch.rand(config.n_neurons, config.n_neurons) < config.sparsity)
        self.register_buffer("adjacency", adj.float())

        self.synapse_state = nn.Parameter(torch.randn(config.n_neurons, config.n_neurons) * 0.01)
        self.local_proj = nn.Linear(config.n_neurons, config.n_neurons, bias=False)

        self.norm = nn.LayerNorm(config.n_neurons)
        self.readout = nn.Linear(config.n_neurons, config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.num_labels)

    def forward(self, idx, cls_labels=None):
        B, T = idx.shape
        acts = torch.zeros(B, self.config.n_neurons, device=idx.device)

        for t in range(T):
            emb = self.token_embed(idx[:, t])
            acts = F.relu(emb)

            propagated = torch.matmul(
                self.adjacency * self.synapse_state,
                acts.T
            ).T

            acts = F.relu(self.local_proj(acts + 0.5 * propagated))

        pooled = self.norm(acts)
        hidden = F.relu(self.readout(pooled))
        cls_logits = self.classifier(hidden)

        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(cls_logits, cls_labels)

        return {
            "cls_logits": cls_logits,
            "loss": loss
        }
