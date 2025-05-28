import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import ResnetBlockFC
from performer_pytorch import Performer
from im2mesh.onet.models import decoder
from im2mesh.layers import (
    CResnetBlockConv1d,
)

# ------------------------------------------------------------
class PEncoder(nn.Module):
    def __init__(self, p_dim=3, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, p):
        B, T, _ = p.shape
        p_flat = p.reshape(B*T, -1)

        feat = self.net(p_flat)   # (B*T, hidden_size)
        feat = feat.view(B, T, -1)
        p_emb = feat.mean(dim=1)  # (B, hidden_size)
        return p_emb

class LatentFlow(nn.Module):
    def __init__(self, z_dim=0, c_dim=256, p_dim=3, hidden_size=256):
        super().__init__()
        self.z_dim = c_dim
        self.p_encoder = PEncoder(p_dim=p_dim, hidden_size=hidden_size)
        self.c_encoder = nn.Linear(c_dim, hidden_size)

        self.flow_net = nn.Sequential(
            nn.Linear(self.z_dim + hidden_size*2 + 1, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.z_dim)
        )

    def forward(self, z, p, c, t):
        p_emb = self.p_encoder(p)      # (B, hidden_size)
        c_emb = self.c_encoder(c)      # (B, hidden_size)

        B = z.size(0)
        t_tensor = torch.full((B, 1), float(t), device=z.device)

        flow_inp = torch.cat([z, p_emb, c_emb, t_tensor], dim=-1)
        dz = self.flow_net(flow_inp)  # (B, z_dim=256)
        return dz


class LatentFlowDecoder(nn.Module):
    def __init__(self, dim=3, z_dim=0, c_dim=256, hidden_size=256, n_steps=5):
        super().__init__()
        self.flow_z_dim = c_dim
        self.n_steps = n_steps

        self.flow = LatentFlow(
            z_dim=self.flow_z_dim,  
            c_dim=c_dim, 
            p_dim=dim,
            hidden_size=hidden_size
        )
        self.decoder = decoder.DecoderCBatchNorm(
            dim=dim,
            z_dim=c_dim,   # decoder z can be used only in flow decoder (even is z_dim=0 in yaml)
            c_dim=c_dim,
            hidden_size=hidden_size
        )
    def forward(self, p, z, c):
        B = p.size(0)
        device = p.device
        internal_z = torch.randn(B, self.flow_z_dim, device=device)
        
        # 2) Euler step
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            dz = self.flow(internal_z, p, c, t=step)
            internal_z = internal_z + dt * dz

        logits = self.decoder(p, internal_z, c)
        return logits
# ------------------------------------------------------------

class PerformerDecoder(nn.Module):
    def __init__(self, dim=3, z_dim=0, c_dim=128,hidden_size=256, performer_hidden_size=64,leaky=False, 
                 legacy=False,num_blocks=3,heads=2,ff_mult=4):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.hidden_size = hidden_size
        self.performer_hidden_size = performer_hidden_size

        if self.z_dim != 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        self.proj_in = nn.Linear(hidden_size, performer_hidden_size)
        self.proj_out = nn.Linear(performer_hidden_size, hidden_size)

        self.performer = Performer(
            dim=performer_hidden_size, 
            depth=num_blocks,    
            heads=heads,     
            dim_head=(performer_hidden_size // heads),
            ff_mult=ff_mult,          
            causal=False      
        )
        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)  # [B, D, T]
        batch_size, D, T = p.size()

        net = self.fc_p(p)  # [B, hidden_size, T]

        if self.z_dim != 0:
            z_feat = self.fc_z(z)               # [B, hidden_size]
            z_feat = z_feat.unsqueeze(2)        # [B, hidden_size, 1]
            z_feat = z_feat.expand(-1, -1, T)   # [B, hidden_size, T]
            net = net + z_feat

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        net = net.transpose(1, 2)  # [B, T, hidden_size]
        net = self.proj_in(net)    # [B, T, performer_dim]
        net = self.performer(net)  # [B, T, performer_dim]
        net = self.proj_out(net)   # [B, T, hidden_size]
        net = net.transpose(1, 2)  # [B, hidden_size, T]

        out = self.fc_out(self.actvn(net))  # [B, 1, T]
        out = out.squeeze(1)                # [B, T]

        return out