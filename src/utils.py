import math
import numpy as np
import torch

from typing import Tuple, Optional

def to_device(obj, device):
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(list(obj), device))
    elif isinstance(obj, dict):
        retval = dict()
        for key, value in obj.items():
            retval[to_device(key, device)] = to_device(value, device)
        return retval 
    elif hasattr(obj, "to"): 
        return obj.to(device)
    else:
        return obj

def positional_encoding(x, L):
    out = [x]
    for j in range(L): #??? to include x then should be L_pos*6 + 3 but in paper, dim is 60 with L=10 ...????
        out.append(torch.sin(2 ** j * x))
        out.append(torch.cos(2 ** j * x))
    return torch.cat(out, dim=1)

def generate_rays(w:int, h:int, f, camera_pose:Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    """ generate the ray's origin and direction (normalized)"""
    # origin is zeros
    ray_o = np.zeros(((w*h), 3))

    u, v = np.arange(w), np.arange(h)
    # assert (u.shape[0], v.shape[0]) == (w, h)
    u, v = np.meshgrid(u, v)
    # print(u.shape, v.shape) # (h, w) (h, w)

    # move 'u, v' to the center. and reflect along axis y.
    # [u, v, f]
    ray_d = np.stack(((u - w/2), -(v - h/2), -np.ones((h,w))*f), axis=-1)
    if camera_pose is not None:
        ray_d = (camera_pose[:3, :3] @ ray_d[..., None]).squeeze(-1)
    # normalize it 
    ray_d = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True)
    # assert ray_d.shape == (h, w, 3)

    if camera_pose is not None:
        ray_o += camera_pose[:3, 3]
    
    return (ray_o, ray_d.reshape(-1, 3))

def calculate_focal_length(fov_in_rad, size_1_dim):
    return size_1_dim / (2 * math.tan(fov_in_rad / 2))




                       