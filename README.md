### FM-ONet: Occupancy Networks with Flowmatching

conda env create -f fm_onet.yaml
conda activate fm_onet

### Train
## Adjust --nproc_per_node based on visible CUDA devices.
## Note: Before training, modify encoder and decoder settings in configs/img/onet.yaml.
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/img/onet.yaml


### Generate (Inference)
## Note: This step uses a pretrained model checkpoint specified in configs/demo.yaml.
python generate.py configs/demo.yaml