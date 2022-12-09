# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code is based on: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.modeling.layers import MultiHeadAttention
from ppdet.modeling.initializer import zeros_, normal_
from ppdet.core.workspace import register

from .models import ModifiedResNet, VisionTransformer, TextEncoder


@register
class CLIP(nn.Layer):
    __inject__ = ['image_encoder', 'text_encoder']

    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.visual = image_encoder
        self.text = text_encoder
        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.weight.shape[0]**-0.5
                normal_(self.visual.attnpool.q_proj.weight, std=std)
                normal_(self.visual.attnpool.k_proj.weight, std=std)
                normal_(self.visual.attnpool.v_proj.weight, std=std)
                normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                    self.visual.layer1, self.visual.layer2, self.visual.layer3,
                    self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        zeros_(param)

        normal_(self.text.token_embedding.weight, std=0.02)
        normal_(self.text.positional_embedding, std=0.01)
        proj_std = (self.text.transformer.width**-0.5) * (
            (2 * self.text.transformer.layers)**-0.5)
        attn_std = self.text.transformer.width**-0.5
        fc_std = (2 * self.text.transformer.width)**-0.5
        for block in self.text.transformer.resblocks:
            normal_(block.attn.in_proj_weight, std=attn_std)
            normal_(block.attn.out_proj.weight, std=proj_std)
            normal_(block.mlp.c_fc.weight, std=fc_std)
            normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text.text_projection is not None:
            normal_(
                self.text.text_projection.weight,
                std=self.text.transformer.width**-0.5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, normalize):
        image_features = self.visual(image.cast(self.dtype))
        if normalize:
            image_features /= image_features.norm(axis=1, keepdim=True)
        return image_features

    def encode_text(self, text, normalize):
        text_features = self.text(text.cast(self.dtype))
        if normalize:
            text_features /= text_features.norm(axis=1, keepdim=True)
        return text_features

    def forward(self, image, text, normalize=True):
        image_features = text_features = None
        if image is not None:
            image_features = self.encode_image(image, normalize)

        if text is not None:
            text_features = self.encode_text(text, normalize)

        return image_fetaures, text_features
