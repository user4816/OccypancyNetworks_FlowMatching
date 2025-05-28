# from im2mesh import icp
import torch
import torch.distributed as dist
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class BaseTrainer(object):
    ''' Base trainer class.
    '''
    def evaluate(self, val_loader):
        ''' Performs an evaluation in a DDP setting, averaging results across all ranks.

        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.model.eval()

        # [CHANGED/ADDED] local sums and counts
        local_sum = defaultdict(float)
        local_count = defaultdict(float)

        # Iterate over validation set (each rank gets a portion)
        for data in tqdm(val_loader, desc='Evaluating'):
            eval_step_dict = self.eval_step(data)
            # Accumulate
            for k, v in eval_step_dict.items():
                local_sum[k] += v
                local_count[k] += 1

        # [CHANGED/ADDED] Now we use all_reduce to sum up across ranks
        device = next(self.model.parameters()).device

        final_dict = {}
        for k in local_sum.keys():
            sum_t = torch.tensor([local_sum[k]], device=device)
            count_t = torch.tensor([local_count[k]], device=device)

            # Sum over all ranks
            dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

            # Compute average
            if count_t.item() > 0:
                final_dict[k] = sum_t.item() / count_t.item()
            else:
                final_dict[k] = 0.

        return final_dict

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError
