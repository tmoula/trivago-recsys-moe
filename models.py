import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAwareSASRec(nn.Module):
    def __init__(self, num_refs, num_actions, max_seq_len=20, hidden_dim=128, num_heads=4, dropout_rate=0.3):
        super().__init__()
        self.ref_embedding      = nn.Embedding(num_refs + 1,    hidden_dim, padding_idx=0)
        self.action_embedding   = nn.Embedding(num_actions + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len,     hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.layer_norm  = nn.LayerNorm(hidden_dim)
        self.dropout     = nn.Dropout(dropout_rate)
        self.head        = nn.Linear(hidden_dim, num_refs + 1)

    def _mask(self, s, device):
        m = torch.triu(torch.ones(s, s, device=device), diagonal=1)
        return m.masked_fill(m == 1, float('-inf'))

    def forward(self, ref_ids, action_ids):
        b, s = ref_ids.size()
        pos  = torch.arange(s, device=ref_ids.device).unsqueeze(0).expand(b, -1)
        x    = (self.ref_embedding(ref_ids) +
                self.action_embedding(action_ids) +
                self.position_embedding(pos))
        x    = self.dropout(self.layer_norm(x))
        out  = self.transformer(x, mask=self._mask(s, ref_ids.device))
        return self.head(out[:, -1, :])


class PersonaConditionedMoEBridge(nn.Module):
    def __init__(self, source_dim=256, target_dim=256, num_personas=7):
        super().__init__()
        self.gating_network = nn.Sequential(
            nn.Linear(num_personas, num_personas * 4),
            nn.ReLU(),
            nn.Linear(num_personas * 4, num_personas)
        )
        self.experts = nn.Parameter(torch.randn(num_personas, target_dim, source_dim))
        nn.init.xavier_uniform_(self.experts)

    def forward(self, H_s, P):
        gate_weights  = F.softmax(self.gating_network(P), dim=-1)
        expert_outputs = torch.einsum('eij,bj->bei', self.experts, H_s)
        H_t_hat = torch.matmul(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)
        return H_t_hat