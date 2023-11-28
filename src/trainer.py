import time
import torch
import numpy as np
import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm as nb_tqdm
from tqdm import tqdm

from typing import Optional, Tuple



from .utils import to_device

class NeRFTrainer(object):
    def __init__(self,
                 model,
                 Renderer,
                 optimizer_builder,
                 loss_func,
                 lr_schedule_builder=None,
                 checkpoint_dir="model_cp"):
        # 
        self.model = model
        self.renderer = Renderer(model)

        self.optimizer_builder = optimizer_builder
        self.loss_func = loss_func
        self.lr_schedule_builder = lr_schedule_builder
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.tqdm = tqdm

        # For test render
        self.test_ray_o = None
        self.test_ray_d = None
        self.b_render_test_render = False
        self.i_render_period = 1

        # reset
        self.reset()

    def set_render_rays(self, ray_o:Optional[torch.Tensor], ray_d:Optional[torch.Tensor]):
        self.test_ray_o = ray_o
        self.test_ray_d = ray_d

    def plot_render(self, save:bool, render_period:int):
        self.b_render_test_render = save
        self.i_render_period = render_period

    def set_tqdm_for_notebook(self, notebook_tqdm=False):
        self.tqdm = nb_tqdm if notebook_tqdm else tqdm

    def _init(self):
        self.result = {}

        record_keys = ["epoch", "total time", "train loss"]

        for item in record_keys:
            self.result[item] = []

    def reset(self):
        self.total_time = 0
        self.last_epoch = 0
        self.optimizer = self.optimizer_builder(self.model)
        self.lr_schedule = None if self.lr_schedule_builder is None else self.lr_schedule_builder()
        self.result = {}

    def load_data(self, filepath):
        self.reset()

        data = torch.load(filepath)
        if data.get('epoch') is not None:
            self.last_epoch = data.get('epoch')
        if data.get('result') is not None:
            self.result = data.get('result')
        if self.result.get('total time') is not None and len(self.result['total time'])!=0:
            self.total_time = self.result['total time'][-1]
        self.model.load_state_dict(data.get('model_state_dict'))
        self.optimizer.load_state_dict(data.get('optimizer_state_dict'))
        if self.lr_schedule is not None:
            self.lr_schedule.load_state_dict(data.get('rl_schedule_state_dict'))


    def save_data(self, filename):
        torch.save({
            'epoch': self.last_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rl_schedule_state_dict': None if self.lr_schedule is None else self.rl_schedule.state_dict(),
            'result' : self.result,
            }, os.path.join(self.checkpoint_dir, filename))

    def get_result(self):
        return pd.DataFrame.from_dict(self.result)


    @staticmethod
    def batchify_rays(model, batch_size, ray_o:torch.Tensor, ray_d:torch.Tensor):
        return torch.cat([model(ray_o[i:i+batch_size], ray_d[i:i+batch_size]).cpu().detach() for i in range(0, ray_o.shape[0], batch_size)], dim=0)

    def run_epoch(self, data_loader, tn:float, tf:float, n_bins:int, device, desc=None, prefix=""):
        losses = []

        # measure the time
        start = time.time()
        for batch in self.tqdm(data_loader, desc=desc, leave=False):
            o = batch[..., 0].to(device)
            d = batch[..., 1].to(device)
            target = batch[..., 2].to(device)

            y_hat = self.renderer(o, d, tn, tf, n_bins, device)
            
            loss = self.loss_func(y_hat, target)

            # only when training
            if self.model.training:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses.append(loss.item())

        end = time.time()

        self.result[prefix + " loss"].append(np.mean(losses))

        return end-start

    def train(self, train_loader, img_size:Tuple[int, int], tn:float, tf:float, n_bins:int, epochs=10, device='cpu', reset=True, cp_filename=None, cp_period=10, print_progress=False, one_time=False):
        ##
        # initialize
        if reset: self.reset()

        # init result
        if len(self.result) == 0 or reset:
            self._init()

        # set device
        is_cuda = False
        if type(device) == torch.device:
            is_cuda = device.type.startswith('cuda')
        elif type(device) == str:
            is_cuda = device.startswith('cuda')

        if is_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        if self.test_ray_o is not None:
            self.test_ray_o = self.test_ray_o.to(device)

        if self.test_ray_d is not None:
            self.test_ray_d = self.test_ray_d.to(device)

        self.model.to(device)

        for epoch in self.tqdm(range(self.last_epoch + 1, self.last_epoch + 1 + epochs), desc="Epoch"):
            self.model = self.model.train()

            self.total_time += self.run_epoch(train_loader, tn, tf, n_bins, device, prefix="train", desc="training")

            self.result["epoch"].append( epoch )
            self.result["total time"].append( self.total_time )


            self.model = self.model.eval()
            if self.b_render_test_render and self.test_ray_o is not None and self.test_ray_d is not None and epoch % self.i_render_period == 0:
                with torch.no_grad():
                    out = self.batchify_rays(lambda ray_o, ray_d: self.renderer(ray_o, ray_d, tn, tf, n_bins, device), 4096, self.test_ray_o, self.test_ray_d)
                    plt.imshow(out.numpy().reshape(img_size[1], img_size[0], 3))
                    plt.show()

            if self.lr_schedule is not None:
                self.lr_schedule.step()

            self.last_epoch = epoch

            if print_progress:
                total_secs = int(self.total_time)
                print(f"Epoch {epoch} - loss : {self.result['train loss'][-1]}, time : {total_secs//60}:{total_secs%60}")

            if cp_filename is not None and epoch%cp_period == 0:
                self.save_data(cp_filename.format(epoch))
            
            if one_time:
                break

        return self.get_result()