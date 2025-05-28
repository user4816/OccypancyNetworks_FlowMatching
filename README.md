## What's New
Based on the original Texture Fields implementation(https://github.com/autonomousvision/texture_fields), we introduce enhancements and adjustments to improve model performance by adding Transforemr architecture in encoder/decoder. Users can modify encoder and decoder configurations in the provided YAML files to achieve optimized texture generation results. Our modifications allow clearer, higher-quality outputs for diverse single-view reconstruction tasks.

## Installation
conda env create -f enh_textfield.yaml
conda activate Enh_textfield

## Reproduction(Demo)
python generate.py configs/singleview/texfields/car_demo.yaml

## Result
- Demo result can be check in out>demo.
- Generate result using our pretrained model.
- Checkpoint path is defined in 'configs/singleview/texfields/car_demo.yaml'.


![1](https://github.com/user-attachments/assets/03f88f1a-8d4a-41d1-a42e-95cb8da30030)



