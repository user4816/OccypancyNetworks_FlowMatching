import os
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples
    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.decoder_type = type(model.module.decoder).__name__
        else:
            self.decoder_type = type(model.decoder).__name__

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        self.model.eval()
        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs').to(device)

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        with torch.no_grad():
            if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                c = self.model.module.encode_inputs(inputs)
                q_z = self.model.module.infer_z(points, occ, c)
                p0_z = self.model.module.p0_z
                z = q_z.mean
                logits = self.model.module.decoder(points, z, c)
            else:
                c = self.model.encode_inputs(inputs)
                q_z = self.model.infer_z(points, occ, c)
                p0_z = self.model.p0_z
                z = q_z.mean
                logits = self.model.decoder(points, z, c)

            # BCE + KL (z) + flow loss
            loss_occ = F.binary_cross_entropy_with_logits(logits, occ, reduction='none').sum(-1).mean()
            kl = dist.kl_divergence(q_z, p0_z).sum(dim=-1).mean() 
            loss = loss_occ + kl
            eval_dict['loss'] = loss.item()

            # IOU calculation: z=mean -> decode -> threshold
            if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                logits_iou = self.model.module.decoder(points_iou, z, c)
            else:
                logits_iou = self.model.decoder(points_iou, z, c)

            p_out = dist.Bernoulli(logits=logits_iou)
            occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
            occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
            eval_dict['iou'] = iou

        return eval_dict


    def visualize(self, data):
        device = self.device
        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5]*3, [0.5]*3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        with torch.no_grad():
            if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                c = self.model.module.encode_inputs(inputs)
                q_z = self.model.module.infer_z(p, torch.zeros_like(p[..., 0]), c)
                z = q_z.mean if not self.eval_sample else q_z.sample()
                logits = self.model.module.decoder(p, z, c)
            else:
                c = self.model.encode_inputs(inputs)
                q_z = self.model.infer_z(p, torch.zeros_like(p[..., 0]), c)
                z = q_z.mean if not self.eval_sample else q_z.sample()
                logits = self.model.decoder(p, z, c)

            p_r = dist.Bernoulli(logits=logits)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))


    def compute_loss(self, data):
        device = self.device
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs').to(device)

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            c = self.model.module.encode_inputs(inputs)
            q_z = self.model.module.infer_z(points, occ, c)
            p0_z = self.model.module.p0_z
            z = q_z.rsample() 
            logits = self.model.module.decoder(points, z, c)
        else:
            c = self.model.encode_inputs(inputs)
            q_z = self.model.infer_z(points, occ, c)
            p0_z = self.model.p0_z
            z = q_z.rsample()
            logits = self.model.decoder(points, z, c)

        # BCE
        loss_occ = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none'
        ).sum(-1).mean()

        kl = dist.kl_divergence(q_z, p0_z).sum(dim=-1).mean()
        loss = loss_occ + kl
        return loss
