# im2mesh/encoder/vit_encoder.py

import torch.nn as nn
import timm
from im2mesh.common import normalize_imagenet

class ViTEncoder(nn.Module):
    ''' Vision Transformer encoder for image input.

    Args:
        c_dim (int): Output dimension of the latent embedding.
        normalize (bool): Whether to normalize input images.
        model_name (str): ViT model variant.
        pretrained (bool): Use pretrained ViT weights.
    '''

    def __init__(self, c_dim=256, normalize=True, model_name='vit_small_patch16_224', pretrained=True):
        super().__init__()
        self.normalize = normalize
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.fc = nn.Linear(self.vit.embed_dim, c_dim)

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        features = self.vit.forward_features(x)  # [batch, embed_dim]
        out = self.fc(features)
        return out
