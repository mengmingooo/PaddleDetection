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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from .clip import *

__all__ = ['ClipImageTextEmbedder']


@register
class ClipImageTextEmbedder(nn.Layer):
    # This code is based on: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
    def __init__(self, base_model, embed_dim, merge_class_token='drop'):
        super().__init__()
        self.clip = base_model
        self.merge_class_token = merge_class_token
        if self.merge_class_token == 'mul-ln':
            self.merged_class_token = nn.LayerNorm(embed_dim)

    def forward(self, images, texts):
        if texts is not None:
            texts_shape = texts.shape
            if len(texts_shape) > 2:
                texts = texts.reshape(-1, texts_shape[-1])

        if images is not None:
            images = normalize_image(images)

        img_emb, txt_emb = self.clip(images, texts, normalize=False)

        if img_emb is not None:
            if self.merge_class_token == 'drop':
                img_emb = img_emb[:, 1:, :]
            elif self.merge_class_token == 'mul-ln':
                img_emb = img_emb[:, :1, :] * img_emb[:, 1:, :]
                img_emb = self.merged_class_token(img_emb)
            else:
                raise ValueError(
                    f'Unknown merge_class_token: {self.merge_class_token}')

        if txt_emb is not None and len(texts_shape) > 2:
            txt_emb = txt_emb.reshape(texts_shape[:-1] + [-1, ])
        return img_emb, txt_emb
