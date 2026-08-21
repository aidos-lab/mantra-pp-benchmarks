import torch
import torch.nn as nn
import torch.nn.functional as F

def pack_to_padded(X, batch_idx):
    device = X.device
    num_graphs = int(batch_idx.max().item()) + 1
    
    counts = torch.bincount(batch_idx)
    max_nodes = int(counts.max().item())
    
    cum_counts = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)])
    node_idx = torch.arange(X.size(0), device=device) - cum_counts[batch_idx]
    
    X_padded = torch.zeros(num_graphs, max_nodes, X.size(1), device=device, dtype=X.dtype)

    key_padding_mask = torch.ones(num_graphs, max_nodes, device=device, dtype=torch.bool)
    
    X_padded[batch_idx, node_idx] = X
    key_padding_mask[batch_idx, node_idx] = False
    
    return X_padded, key_padding_mask, node_idx

def padded_to_pack(X_padded, batch_idx, node_idx):
    return X_padded[batch_idx, node_idx]


class VectorizedMAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(VectorizedMAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        
        self.mha = nn.MultiheadAttention(embed_dim=dim_V, num_heads=num_heads, batch_first=True)
        
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, batch_idx_Q=None, batch_idx_K=None):
        # 1. Project features
        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V_proj = self.fc_v(K)

        if batch_idx_Q is None or batch_idx_K is None:
            attn_out, _ = self.mha(Q_proj, K_proj, V_proj)
            O = Q_proj + attn_out
        else:
            Q_pad, _, q_node_idx = pack_to_padded(Q_proj, batch_idx_Q)
            K_pad, k_pad_mask, _ = pack_to_padded(K_proj, batch_idx_K)
            V_pad, _, _ = pack_to_padded(V_proj, batch_idx_K)
            
            attn_out_pad, _ = self.mha(Q_pad, K_pad, V_pad, key_padding_mask=k_pad_mask)
            
            O_pad = Q_pad + attn_out_pad
            O = padded_to_pack(O_pad, batch_idx_Q, q_node_idx)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = VectorizedMAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, batch_idx=None):
        return self.mab(X, X, batch_idx, batch_idx)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.num_inds = num_inds
        self.I = nn.Parameter(torch.Tensor(num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = VectorizedMAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = VectorizedMAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, batch_idx):
        num_graphs = int(batch_idx.max().item()) + 1
        
        I_all = self.I.repeat(num_graphs, 1)
        batch_idx_I = torch.arange(num_graphs, device=X.device).repeat_interleave(self.num_inds)

        H = self.mab0(I_all, X, batch_idx_I, batch_idx)
        return self.mab1(X, H, batch_idx, batch_idx_I)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.num_seeds = num_seeds
        self.S = nn.Parameter(torch.Tensor(num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = VectorizedMAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, batch_idx):
        num_graphs = int(batch_idx.max().item()) + 1
        
        S_all = self.S.repeat(num_graphs, 1)
        batch_idx_S = torch.arange(num_graphs, device=X.device).repeat_interleave(self.num_seeds)
        
        pooled = self.mab(S_all, X, batch_idx_S, batch_idx)
        
        return pooled.view(num_graphs, self.num_seeds, -1)


class SetTransformer(nn.Module):
    def __init__(self, in_channels, num_outputs, out_channels,
            num_inds, hidden_channels, num_heads, ln=False):
        super(SetTransformer, self).__init__()
        self.enc0 = ISAB(in_channels, hidden_channels, num_heads, num_inds, ln=ln)
        self.enc1 = ISAB(hidden_channels, hidden_channels, num_heads, num_inds, ln=ln)
        
        self.pma = PMA(hidden_channels, num_heads, num_outputs, ln=ln)
        self.sab0 = SAB(hidden_channels, hidden_channels, num_heads, ln=ln)
        self.sab1 = SAB(hidden_channels, hidden_channels, num_heads, ln=ln)
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, batch):
        X = batch['x'] 
        batch_idx = batch['batch']
        
        X = self.enc0(X, batch_idx)
        X = self.enc1(X, batch_idx)
        
        X = self.pma(X, batch_idx)
        
        X = self.sab0(X)
        X = self.sab1(X)
        
        X = X.mean(dim=1) 
        return self.fc_out(X)
