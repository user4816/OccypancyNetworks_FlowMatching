## What's New
Based on the original Occupancy Networks implementation(https://github.com/autonomousvision/occupancy_networks), we introduce enhancements by integrating Flow Matching into the training pipeline. And also, we added transformer architecture in Encoder and Decoder, so users can customize encoder and decoder configurations through provided YAML files to optimize occupancy prediction results. 

## Installation
conda env create -f fm_onet.yaml
conda activate fm_onet

## Reproduction(Demo)
python generate.py configs/demo.yaml

## Install Meshlab
- sudo apt update
- sudo apt install -y meshlab
- meshlab demo/generation/meshes/02.jpg.off

## Result
- Demo result can be check in demo/generation/meshes.
- Generate result using our pretrained model.
- Checkpoint path is defined in 'configs/demo.yaml'.

![그림1](https://github.com/user-attachments/assets/4689d7df-e6b0-4111-a6b4-821d041301bf)


