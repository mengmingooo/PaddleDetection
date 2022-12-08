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

import numpy as np

import paddle
import paddle.nn.functional as F

IMAGE_MEAN = paddle.to_tensor([0.48145466, 0.4578275, 0.40821073])
IMAGE_STD = paddle.to_tensor([0.26862954, 0.26130258, 0.27577711])


def normalize_image(img):
    return (img - IMAGE_MEAN) / IMAGE_STD


def unnormalize_image(x):
    return x * IMAGE_STD + IMAGE_MEAN


def resize_posemb(posemb, target_size):
    """Resizes position embeddings to new resolution."""
    if target_size == posemb.shape[1]:
        return posemb

    gs_old = int(np.sqrt(posemb.shape[1]))
    gs_new = int(np.sqrt(target_size))

    posemb_tok = None
    if gs_old**2 == posemb.shape[1]:
        posemb_grid = posemb
    elif gs_old**2 == posemb.shape[1] - 1:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[:, 1:]
    else:
        raise ValueError(
            'Posemb shape must be a perfect square (maybe with CLS token), but '
            f'got posemb of shape {posemb.shape}.')

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).transpose(
        [0, 3, 1, 2])
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode='bilinear', align_corners=False)
    posemb_grid = posemb_grid.transpose([0, 2, 3, 1]).reshape(1, gs_new[0] *
                                                              gs_new[1], -1)
    if posemb_tok is not None:
        posemb = paddle.concat([posemb_tok, posemb], axis=1)

    return posemb


def seq2img(original_img, features):
    """Reshapes 1D sequence to 2D image features."""
    if original_img.shape[2] == original_img.shape[3]:
        h = w = int(np.sqrt(features.shape[2]))
    else:
        stride = np.ceil(
            np.sqrt(original_img.shape[2] * original_img.shape[3] /
                    features.shape[2]))
        h = np.ceil(original_img.shape[2] / stride)
        w = np.ceil(original_img.shape[3] / stride)
    return features.reshape([features.shape[0], -1, int(h), int(w)])


def normalized_grid_corner_coordinates(feature_map, padding_mask):
    """Computes normalized xy corner coords from feature_map or padding_mask."""
    # Note 1: it computes not the centers of grid patches, but the patch corner
    # coordinates (for a grid patch from 0 to 0.1, it returns 0.1 not 0.05).
    # Note 2: behavior is quite different for feature_map and padding_mask inputs.
    if padding_mask is None:
        assert len(feature_map.shape) == 4  # [B, C, H, W]
        _, _, h, w = paddle.shape(feature_map)
        shift_x = paddle.arange(1, w + 1)
        shift_y = paddle.arange(1, h + 1)
        shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
        # [H, W, 2]
        xy = paddle.cast(
            paddle.stack(
                [shift_x, shift_y], axis=-1), dtype='float32')
        xy = xy / paddle.concat([w, h])
    else:
        assert len(padding_mask.shape) == 3  # [B, H, W]
        padding_mask = padding_mask.cast(paddle.float32)
        y = paddle.cumsum(padding_mask, axis=1)
        x = paddle.cumsum(padding_mask, axis=2)
        # [B, H, W, 2]
        xy = paddle.stack(
            [x / (x[:, :, -1:] + 1e-6), y / (y[:, -1:] + 1e-6)], axis=-1)

    return xy.reshape(xy.shape[:-3] + [-1, 2])


def compute_box_bias(feature_map, padding_mask, kind='both'):
    """Computes spatial bias for grid."""
    # The box center is biased to its position on the feature grid:
    xy = normalized_grid_corner_coordinates(feature_map, padding_mask)
    xy = paddle.clip(xy, 0.0, 1.0)

    if kind in ['both', 'location']:
        # Unnormalize xy (i.e., apply logit function/sigmoid^-1).
        xy_bias = logit(xy)
    else:
        xy_bias = paddle.zeros_like(xy)

    if kind in ['both', 'size']:
        # The box size is biased to the patch size:
        wh_bias = logit(paddle.full_like(xy_bias, 1.0 / feature_map.shape[-1]))
    else:
        wh_bias = paddle.zeros_like(xy_bias)

    return paddle.concat([xy_bias, wh_bias], axis=-1)


def logit(x, eps=1e-4):
    """Logit (inverse sigmoid) function (https://en.wikipedia.org/wiki/Logit)."""
    return paddle.log(x + eps) - paddle.log1p(-x + eps)
