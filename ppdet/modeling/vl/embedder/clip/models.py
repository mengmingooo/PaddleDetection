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

from ppdet.modeling.initializer import zeros_, normal_
from ppdet.core.workspace import register

from .layers import *

__all__ = ['ModifiedResNet', 'ViT', 'TextEncoder']


@register
class ModifiedResNet(nn.Layer):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()

        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2D(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(
            width // 2, width // 2, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(
            width // 2, width, kernel_size=3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(width)
        self.relu3 = nn.ReLU()
        self.avgpool = nn.AvgPool2D(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2D(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.cast(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


@register
class ViT(nn.Layer):
    def __init__(self,
                 input_resolution,
                 patch_size,
                 width,
                 layers,
                 heads,
                 output_dim=None,
                 stochastic_droplayer_rate=0.0):
        super().__init__()
        self.width = width
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias_attr=False)
        scale = width**-0.5
        self.class_embedding = self.create_parameter(
            shape=[width], attr=ParamAttr(initializer=Normal(std=scale)))
        self.positional_embedding = self.create_parameter(
            shape=[(input_resolution // patch_size)**2 + 1, width],
            attr=ParamAttr(initializer=Normal(std=scale)))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads,
                                       stochastic_droplayer_rate)
        self.ln_post = LayerNorm(width)
        if output_dim is not None:
            self.proj = nn.Linear(self.width, self.output_dim, bias_attr=False)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape([x.shape[0], x.shape[1], -1])
        x = x.transpose([0, 2, 1])
        class_embedding = self.class_embedding.cast(x.dtype) + paddle.zeros(
            [x.shape[0], 1, x.shape[-1]], type=x.dtype)
        x = paddle.concat([class_embedding, x], axis=1)
        x = x + self.positional_embedding.cast(x.dtype)
        x = self.ln_pre(x)
        x = feature = self.transformer(x)
        if self.output_dim is not None:
            x = self.ln_post(x[:, 0, :])
            x = self.proj(x)
        else:
            x = self.ln_post(x)

        return x, feature


@register
class TextEncoder(nn.Layer):
    def __init__(self,
                 embed_dim,
                 context_length,
                 vocab_size,
                 transformer_width,
                 transformer_heads,
                 transformer_layers,
                 stochastic_droplayer_rate=0.0):
        super().__init__()
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            stochastic_droplayer_rate=stochastic_droplayer_rate,
            attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = self.create_parameter(
            shape=[transformer_width, embed_dim],
            attr=ParamAttr(initializer=Constant(0.0)))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Linear(
            transformer_width, embed_dim, bias_attr=False)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = paddle.full((self.context_length, self.context_length),
                           float("-inf"))
        mask = paddle.triu(mask)
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.cast(x.dtype)
        x = self.transformer(x)
        x = self.ln_final(x).cast(x.dtype)

        # x.shape = [batch_size, text_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        batch_idx = paddle.arange(x.shape(0))
        seq_idx = text.argmax(dim=-1)
        gather_idx = paddle.stack([batch_idx, seq_idx], axis=1)
        x = paddle.gather_nd(x, gather_idx)
        x = self.text_projection(x)

        return x
