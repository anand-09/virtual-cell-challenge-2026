"""
Neural Network Architecture for Single-Cell Perturbation Response:
Context + Gene Perturbation -> 18,533 Transcriptomic Response.

Architecture:
1. Perturbation Embedding: Gene embedding / One-hot projection
2. Context Embedding: Categorical context embedding (A, B, C)
3. Deep Residual MLP Decoder: Predicts single-cell expression shift (Delta)
   or full transcriptomic profile over the 18,533-gene axis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleCellPerturbationMLP(nn.Module):
    def __init__(
        self,
        n_genes: int = 18533,
        n_contexts: int = 3,
        n_perts: int = 301,
        pert_embed_dim: int = 128,
        context_embed_dim: int = 32,
        hidden_dims: list = [512, 1024, 2048],
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_genes = n_genes
        self.pert_embedding = nn.Embedding(n_perts, pert_embed_dim)
        self.context_embedding = nn.Embedding(n_contexts, context_embed_dim)

        in_dim = pert_embed_dim + context_embed_dim

        layers = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.out_head = nn.Linear(curr_dim, n_genes)

    def forward(self, pert_idx: torch.Tensor, context_idx: torch.Tensor, base_control: torch.Tensor = None) -> torch.Tensor:
        """
        pert_idx: (B,)
        context_idx: (B,)
        base_control: (B, 18533) optional baseline control mean expression
        Returns:
            predicted transcriptomic expression (B, 18533)
        """
        p_emb = self.pert_embedding(pert_idx)
        c_emb = self.context_embedding(context_idx)
        x = torch.cat([p_emb, c_emb], dim=-1)
        h = self.backbone(x)
        delta = self.out_head(h)

        if base_control is not None:
            # Predict perturbation response as delta shift from basal state
            return F.relu(base_control + delta)
        return F.relu(delta)

if __name__ == "__main__":
    model = SingleCellPerturbationMLP()
    print("Model initialized successfully!")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test a forward pass
    dummy_pert = torch.tensor([0, 1])
    dummy_ctx = torch.tensor([0, 1])
    out = model(dummy_pert, dummy_ctx)
    print("Forward pass output shape:", out.shape)
