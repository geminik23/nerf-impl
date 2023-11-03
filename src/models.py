import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRFModelRenderer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ray_o:torch.Tensor, ray_d:torch.Tensor, tn:float, tf:float, n_bins:int, device="cpu"):
        n_rays = ray_o.shape[0]
        t = torch.linspace(tn, tf, n_bins).to(device)

        p = ray_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * ray_d.unsqueeze(1)  # [n_rays, 1, 3] + [1, n_bins, 1] * [n_rays, 1, 3] => [n_rays, n_bins, 3]    
        assert p.shape == (n_rays, n_bins, 3)
        
        # Compute internal transmission
        # T = np.exp(-distance*sigma_a)
        # need to reshpae p to [n_rays*n_bins, 3] and ray_d to [n_rays*n_bins, 3]
        c, d = self.model(p.reshape(-1, 3), ray_d.expand(n_bins, n_rays, 3).transpose(0, 1).reshape(-1, 3))
        c = c.reshape(n_rays, n_bins, 3)
        d = d.reshape(n_rays, n_bins)
        
        delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10]).to(device))) # [(n_bins-1) + 1]
        # [n_bins, 1] * [n_rays, n_bins, 1]
        T = torch.exp(-delta.unsqueeze(0)*d) # [n_rays, n_bins]
        alpha = 1-T
        
        ## Calculate the accumulated transmittance
        acc_t = torch.cumprod(T, 1)
        # acc_t[:, 1:] = acc_t[:, :-1]
        # acc_t[:, 0] = 1.0
        acc_t = torch.cat((torch.ones((acc_t.shape[0], 1), device=device), acc_t[:, :-1]), dim=1)
        # print(acc_t.shape)
        # weight = acc_t * alpha #[n_rays, n_bins, 1]
        # print(weight.shape) 

        i_weight = (acc_t * alpha).unsqueeze(-1) # [n_rays, n_bins, 1]
        return (i_weight * c).sum(1)
        # return (weight.unsqueeze(-1) * c).sum(1) + 1 - weight.sum(-1)


# referred Appendix A in original paper.
class NaiveNeRFModel(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden_dim=256):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir

        # 6 = 3 (dim) * 2 (sin, cos)
        # in paper, 10*6 = 60, 4*6 = 24
        self.net1 = nn.Sequential(nn.Linear(L_pos * 6 + 3, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        
        self.net2 = nn.Sequential(nn.Linear(hidden_dim + L_pos * 6 + 3, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim + 1), nn.ReLU(inplace=True))
        
        self.post_net = nn.Sequential(nn.Linear(hidden_dim + L_dir * 6 + 3, hidden_dim // 2), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())
        
    def _positional_encoding(self, x, L):
        out = [x]
        for j in range(L): #??? to include x then should be L_pos*6 + 3 but in paper, dim is 60 with L=10 ...????
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
        
    def forward(self, pos, d):
        # positional encoding
        gamma_x = self._positional_encoding(pos, self.L_pos) # [batch_size, L_pos * 6 + 3] 
        gamma_d = self._positional_encoding(d, self.L_dir) # [batch_size, L_dir * 6 + 3]
        
        h = self.net1(gamma_x) # [batch_size, hidden_dim]
        h = self.net2(torch.cat((h, gamma_x), dim=1)) # [batch_size, hidden_dim + L_pos * 6 + 3]
        sigma = h[:, -1]
        h = h[:, :-1] # [batch_size, hidden_dim]
        c = self.post_net(torch.cat((h, gamma_d), dim=1))
        return c, sigma
