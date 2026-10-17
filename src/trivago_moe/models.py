"""Model definitions, consolidated from notebooks/polanyilabs_prototype.ipynb.

Pipeline overview:

    Phase 1  SASRecSourceEncoder        retail clicks (Yoochoose)      -> H_s  [B, D_s]
    Phase 2  PersonaConditionedMoEBridge H_s + persona mix P [B, 7]    -> Ĥ_t  [B, D_t]
    Phase 3  FullCrossDomainPipeline     Ĥ_t                           -> hotel logits
    Baseline ContextAwareSASRec          Trivago (hotel, action) seq   -> hotel logits

State-dict keys match the checkpoints produced by the notebook, so published
weights load directly into these classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Additive mask so position i cannot attend to positions > i."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


# ---------------------------------------------------------------------------
# Phase 1: retail source encoder (pre-trained on Yoochoose)
# ---------------------------------------------------------------------------
class SASRecSourceEncoder(nn.Module):
    def __init__(self, num_items: int, max_seq_len: int = 20, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate, activation="relu", batch_first=True)
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, source_clicks: torch.Tensor) -> torch.Tensor:
        """[B, S] padded item ids -> H_s [B, D_s] (state at the last position)."""
        b, s = source_clicks.size()
        pos = torch.arange(s, device=source_clicks.device).unsqueeze(0).expand(b, -1)
        x = self.item_embedding(source_clicks) + self.position_embedding(pos)
        x = self.dropout(self.layer_norm(x))
        out = self.transformer_blocks(x, mask=causal_mask(s, source_clicks.device))
        return out[:, -1, :]


class TrainableSASRecSource(nn.Module):
    """Source encoder + next-item head, used only for Phase 1 pre-training."""

    def __init__(self, encoder: SASRecSourceEncoder):
        super().__init__()
        self.encoder = encoder
        self.prediction_head = nn.Linear(encoder.hidden_dim, encoder.num_items + 1)

    def forward(self, source_clicks: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(self.encoder(source_clicks))


# ---------------------------------------------------------------------------
# Phase 2: persona-conditioned mixture-of-experts bridge
# ---------------------------------------------------------------------------
class PersonaConditionedMoEBridge(nn.Module):
    """Maps retail vectors into travel space with one linear expert per archetype.

    A small MLP turns the persona mixture P into gate weights; the output is the
    gate-weighted sum of the 7 expert projections of H_s.
    """

    def __init__(self, source_dim: int = 256, target_dim: int = 256, num_personas: int = 7):
        super().__init__()
        self.gating_network = nn.Sequential(
            nn.Linear(num_personas, num_personas * 4),
            nn.ReLU(),
            nn.Linear(num_personas * 4, num_personas),
        )
        self.experts = nn.Parameter(torch.randn(num_personas, target_dim, source_dim))
        nn.init.xavier_uniform_(self.experts)

    def forward(self, H_s: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        gate_weights = F.softmax(self.gating_network(P), dim=-1)            # [B, E]
        expert_outputs = torch.einsum("eij,bj->bei", self.experts, H_s)     # [B, E, D_t]
        return torch.matmul(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)


# ---------------------------------------------------------------------------
# Phase 3: travel-side model + joint loss
# ---------------------------------------------------------------------------
class FullCrossDomainPipeline(nn.Module):
    def __init__(self, moe_bridge: PersonaConditionedMoEBridge, num_travel_items: int,
                 hidden_dim: int = 256, max_seq_len: int = 20):
        super().__init__()
        self.moe_bridge = moe_bridge
        self.target_item_embedding = nn.Embedding(num_travel_items + 1, hidden_dim, padding_idx=0)
        self.target_position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True)
        self.target_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.target_layer_norm = nn.LayerNorm(hidden_dim)
        self.travel_prediction_head = nn.Linear(hidden_dim, num_travel_items + 1)

    def forward(self, H_s, P, target_clicks):
        b, s = target_clicks.size()
        device = target_clicks.device
        H_t_hat = self.moe_bridge(H_s, P)
        pos = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        t_embed = self.target_layer_norm(
            self.target_item_embedding(target_clicks) + self.target_position_embedding(pos))
        H_t = self.target_transformer(t_embed, mask=causal_mask(s, device))[:, -1, :]
        # NOTE: logits come from H_t_hat only; H_t is returned but unused by the loss.
        logits = self.travel_prediction_head(H_t_hat)
        return logits, H_t_hat, H_t


class MultiTaskCrossDomainLoss(nn.Module):
    """Next-item cross-entropy + λ · (match in-batch similarity of Ĥ_t to that of P)."""

    def __init__(self, lambda_weight: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, logits, targets, H_t_hat, P):
        L_rec = self.ce(logits, targets)
        norm_H = F.normalize(H_t_hat, p=2, dim=-1)
        sim_H = norm_H @ norm_H.T / self.temperature
        norm_P = F.normalize(P, p=2, dim=-1)
        sim_P = norm_P @ norm_P.T
        L_cl = F.mse_loss(F.softmax(sim_H, dim=-1), F.softmax(sim_P, dim=-1))
        return L_rec + self.lambda_weight * L_cl


# ---------------------------------------------------------------------------
# Trivago-only baseline (the model behind the reported HR@10)
# ---------------------------------------------------------------------------
class ContextAwareSASRec(nn.Module):
    def __init__(self, num_refs: int, num_actions: int, max_seq_len: int = 20,
                 hidden_dim: int = 128, num_heads: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.ref_embedding = nn.Embedding(num_refs + 1, hidden_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(num_actions + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(hidden_dim, num_refs + 1)

    def forward(self, ref_ids: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        b, s = ref_ids.size()
        pos = torch.arange(s, device=ref_ids.device).unsqueeze(0).expand(b, -1)
        x = self.ref_embedding(ref_ids) + self.action_embedding(action_ids) + self.position_embedding(pos)
        x = self.dropout(self.layer_norm(x))
        out = self.transformer(x, mask=causal_mask(s, ref_ids.device))
        return self.head(out[:, -1, :])
