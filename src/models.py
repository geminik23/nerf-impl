import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import positional_encoding


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
        color = (i_weight * c).sum(1)
        return color
        # return color + 1 - i_weight.sum(-1)


##
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
        

    def forward(self, pos, d):
        # positional encoding
        gamma_x = positional_encoding(pos, self.L_pos) # [batch_size, L_pos * 6 + 3] 
        gamma_d = positional_encoding(d, self.L_dir) # [batch_size, L_dir * 6 + 3]
        
        h = self.net1(gamma_x) # [batch_size, hidden_dim]
        h = self.net2(torch.cat((h, gamma_x), dim=1)) # [batch_size, hidden_dim + L_pos * 6 + 3]
        sigma = h[:, -1]
        h = h[:, :-1] # [batch_size, hidden_dim]
        c = self.post_net(torch.cat((h, gamma_d), dim=1))
        return c, sigma



##
# from the FastNeRF paper
class FastNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, D=8, hidden_dim_pos=384, hidden_dim_dir=128):
        """default value is from 'Implementation' section in paper"""
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        self.D = D

        # 7 hidden layers
        self.F_pos = nn.Sequential(nn.Linear(L_pos * 6 + 3, hidden_dim_pos), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_pos, 3 * D + 1), nn.ReLU(inplace=True))


        # 3 hidden layers
        self.F_dir = nn.Sequential(nn.Linear(L_dir * 6 + 3, hidden_dim_dir), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim_dir, D) )

        
    def forward(self, pos, d):
        # positional encoding
        gamma_x = positional_encoding(pos, self.L_pos) # [batch_size, L_pos * 5 + 3] 
        gamma_d = positional_encoding(d, self.L_dir) # [batch_size, L_dir * 5 + 3]

        h = self.F_pos(gamma_x) # [batch_size, 2*D + 1]
        uvw = torch.sigmoid(h[:, :-1])  # [ batch_size, 3*D] : value range is in [0, 1]
        sigma = h[:, -1]

        # inner-product between uvw and beta
        # beta is weights so apply the softmax 
        beta = torch.softmax(self.F_dir(gamma_d), -1) # [batch_size, D] 

        # sum(beta [batch_size, 0, D] * uvw[batch_size, 3, D]) -> [batch_size, 3]
        c = (beta.unsqueeze(1) * uvw.reshape(uvw.shape[0], 3, self.D)).sum(-1)
        c = torch.bmm(uvw.reshape(uvw.shape[0], 3, self.D),beta.unsqueeze(-1)).squeeze(-1)
        return c, sigma


