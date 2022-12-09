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
from ppdet.modeling.losses.iou_loss import GIoULoss
from ppdet.modeling.transformers import bbox_cxcywh_to_xyxy, sigmoid_focal_loss

__all__ = ['OWLViTLoss']


@register
class OWLViTLoss(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['HungarianMatcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 normalization='per_example',
                 loss_coeff=None,
                 use_focal_loss=None,
                 alpha=None,
                 gamma=None):
        super().__init__()
        self.giou_loss = GIoULoss()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = matcher.matcher_coeff if loss_coeff is None else loss_coeff
        self.use_focal_loss = matcher.use_focal_loss if use_focal_loss is None else use_focal_loss
        self.alpha = matcher.alpha if alpha is None else alpha
        self.gamma = matcher.gamma if gamma is None else gamma
        assert normalization in [
            'per_example', 'global'
        ], f'{normalization} should be in [pre_example, global]'
        self.normalization = normalization

    def _get_loss_class(self, logits, gt_class, match_indices):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        target_label = paddle.full(
            logits.shape[:2], self.num_classes, dtype='int64')
        bs, num_query_objects = target_label.shape
        if sum(len(a) for a in gt_class) > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]

        if self.use_focal_loss:
            loss_cls = F.sigmoid_focal_loss(
                logits,
                target_label,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction='none')
        else:
            loss_cls = F.cross_entropy(logits, target_label, reduction='none')

        return loss_cls.sum(axis=[1, 2])

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices):
        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        src_box = bbox_cxcywh_to_xyxy(src_bbox)
        target_bbox = bbox_cxcywh_to_xyxy(target_bbox)
        loss_bbox = F.l1_loss(src_bbox, target_bbox, reduction='none')
        loss_giou = self.giou_loss(src_bbox, target_bbox)
        return loss_bbox.sum(axis=1), loss_giou.sum(axis=1)

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def forward(self, head_outs, gt_meta):
        logits, boxes = head_outs
        gt_class, gt_bbox = gt_meta['gt_class'], gt_meta['gt_bbox']
        match_indices = self.matcher(boxes.detach(),
                                     logits.detach(), gt_bbox, gt_class)
        loss_cls = self._get_loss_class(logits, gt_class, match_indices)
        loss_bbox, loss_giou = self._get_loss_bbox(boxes, gt_bbox,
                                                   match_indices)

        num_gts = paddle.to_tensor([len(a) for a in gt_class])
        if self.normalization == 'per_example':
            num_gts = paddle.clip(num_gts, min=1)
            loss_cls = (loss_cls / num_gts).mean()
            loss_bbox = (loss_bbox / num_gts).mean()
            loss_giou = (loss_giou / num_gts).mean()
            # normalize_fn = lambda x : (x / num_gts).mean()
        else:
            num_gts = paddle.distributed.all_reduce(num_gts)
            num_gts = paddle.clip(
                num_gts / paddle.distributed.get_world_size(), min=1)
            loss_cls = loss_cls.sum() / num_gts
            loss_bbox = loss_bbox.sum() / num_gts
            loss_giou = loss_giou.sum() / num_gts
            # normalize_fn = lambda x: x.sum() / num_gts

        # loss_cls, loss_box, loss_giou = [normalize_fn(l) for l in [loss_cls, loss_box, loss_giou]]
        loss = self.loss_coeff['class'] * loss_cls + \
               self.loss_coeff['bbox'] * loss_bbox + \
               self.loss_coeff['giou'] * loss_giou

        return {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
