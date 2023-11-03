import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from src.utils import generate_rays

class BaseObject(ABC):
    @abstractmethod
    def integrate(self, rays:torch.Tensor):
        pass

class Camera:
    def __init__(self, whf:tuple[int, int, float], matrix:Optional[torch.Tensor]=None, device='cpu'):
        # super().__init__()
        if matrix is None:
            matrix = torch.tensor([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.], [0., 0., 1.0, 0.], [0., 0., 0., 1.0]]).reshape([4,4]).float().to(device)
        else:
            assert matrix.shape == (4, 4)
        self.whf = whf
        self.matrix = matrix.to(device)

    
    def generate_rays(self, device):
        ray_o, ray_d = generate_rays(*self.whf)
        ray_o = torch.from_numpy(ray_o).float().to(device)
        ray_d = torch.from_numpy(ray_d).float().to(device)
        
        return ray_o + self.matrix[:3, 3], (self.matrix[:3, :3] @ ray_d.unsqueeze(-1)).squeeze(-1)



# only consider the internal transmittance of volume. (no light)
class VolumetricRenderer(nn.Module):
    def __init__(self, camera:Camera, obj:BaseObject):
        super().__init__()
        self.camera = camera
        self.obj:BaseObject = obj

    def forward(self, ray_o, ray_d, tn:float, tf:float, n_bins:int, device='cpu'):
        # !! did not use a stratified sampling in paper.
        t = torch.linspace(tn, tf, n_bins).to(device)

        p = ray_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * ray_d.unsqueeze(1) # [n_rays, 1, 3] + [1, n_bins, 1] * [n_rays, 1, 3] => [n_rays, n_bins, 3]    
        # assert p.shape == (ray_o.shape[0], n_bins, 3)
        
        #
        # Compute internal transmission
        # T = np.exp(-distance*sigma_a)
        # color and density of d though rays
        c, d = self.obj.integrate(p)
        delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10]).to(device))) # [(n_bins-1) + 1]
        # [n_bins, 1] * [n_rays, n_bins, 1]
        
        T = torch.exp(-delta.unsqueeze(0)*d) # [n_rays, n_bins]
        alpha = 1-T
        
        ## Calculate the accumulated transmittance
        acc_t = torch.cumprod(T, 1)
        # acc_t[:, 1:] = acc_t[:, :-1]
        # acc_t[:, 0] = 1.0
        acc_t = torch.cat((torch.ones((acc_t.shape[0], 1), device=device), acc_t[:, :-1]), dim=1)
        
        i_weight = (acc_t * alpha).unsqueeze(-1) # [n_rays, n_bins, 1]        
        return (i_weight * c).sum(1)
        
    def generate_rays(self, device='cpu'):
        ray_o, ray_d = self.camera.generate_rays(device) # [n_rays, 3], [n_rays, 3]
        return ray_o, ray_d
        
    def render(self, tn:float, tf:float, n_bins:int, device='cpu'):
        # create the rays
        ray_o, ray_d = self.generate_rays(device)
        return self.forward(ray_o, ray_d, tn, tf, n_bins, device)