import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

import torch
import torch.optim as optim
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter
import numpy as np
import argparse
import time
import matplotlib; matplotlib.use('Agg')
import random
import torch.backends.cudnn as cudnn

#seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO


parser = argparse.ArgumentParser(description='Train a 3D reconstruction model.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds with exit code 2.')
parser.add_argument('--local_rank', type=int, default=0, 
                    help='local rank for DistributedDataParallel')

args = parser.parse_args()

cfg = config.load_config(args.config, 'configs/default.yaml')


dist.init_process_group(backend='nccl', init_method='env://')

if args.local_rank == 0:
    wandb.login(key="0ab7a56f949924bd68ef4e0cea23f8cd23b19636")    
    wandb.init(
        project="Coursework_CS570",
        job_type="train",
        sync_tensorboard=False,
        settings=wandb.Settings(_disable_stats=True),
        config=cfg
    )

torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

t0 = time.time()

out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be either maximize or minimize.')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

train_sampler = torch.utils.data.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=4,
    sampler=train_sampler, 
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=10,
    num_workers=4,
    shuffle=False,  
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn
)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=12, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
data_vis = next(iter(vis_loader))

model = config.get_model(cfg, device=device, dataset=train_dataset)
model = model.to(device)


model = DDP(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)

if torch.cuda.is_available():
    print(f"[Rank {dist.get_rank()}] Running on {torch.cuda.get_device_name(args.local_rank)}")
else:
    print("[Rank 0] CPU Mode Activated.")

npoints = 1000
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get('loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

logger = SummaryWriter(os.path.join(out_dir, 'logs'))

print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

train_start_time = time.time()
while True:
    epoch_it += 1

    train_sampler.set_epoch(epoch_it)

    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch)

        if args.local_rank == 0:
            wandb.log({"train/loss": loss}, step=it)

        logger.add_scalar('train/loss', loss, it)

        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f' % (epoch_it, it, loss))

        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            trainer.visualize(data_vis)

        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)

        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save(f'model_{it}.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)

        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                    % (model_selection_metric, metric_val))

            if args.local_rank == 0:
                wandb.log({f"val/{k}": v for k, v in eval_dict.items()}, step=it)

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)

        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)
            total_training_time = time.time() - train_start_time
            print(f"🎉 Total training time: {total_training_time / 60:.2f} minutes.")

            if args.local_rank == 0:
                wandb.log({"total_training_time_minutes": total_training_time / 60})
                wandb.finish()

            exit(3)
