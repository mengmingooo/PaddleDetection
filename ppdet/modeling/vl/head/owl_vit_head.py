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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.modeling.ops import get_act_fn

from ..utils import compute_box_bias

__all__ = ['PredictorMLP', 'ClassPredictor', 'OWLViTHead']


@register
class PredictorMLP(nn.Layer):
    """FFN block for predicting continuous outputs, e.g. bounding box coordinates.  

    Attributes:
      out_dim: Size of output of this mlp.
      num_layers: Number of layers.
      mlp_dim: Size of hidden dimension of dense layers.
      hidden_activation: Activation function of hidden layers.
      out_activation: Activation of the output.
      dtype: Data type, e.g. jnp.float32.

    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_layers,
                 mlp_dim,
                 hidden_activation,
                 out_activation=None):
        super().__init__()

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, mlp_dim))
            in_dim = mlp_dim

        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.LayerList(layers)
        self.num_layers = num_layers
        self.hidden_activation = get_act_fn(hidden_activation)
        self.out_activation = get_act_fn(out_activation)

    def forward(self, inputs):
        x = inputs
        for _ in range(self.num_layers - 1):
            x = self.mlp[i](x)
            x = self.hidden_activation(x)

        x = self.mlp[-1](x)
        x = self.out_activation(x)

        return x


@register
class ClassPredictor(nn.Layer):
    """Open-vocabulary instance class predictor."""

    def __init__(self, in_dim, out_dim, normalize):
        super().__init__()
        self.normalize = normalize
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)
        self.logit_shift = nn.Linear(in_dim, 1)
        self.logit_scale = nn.Linear(in_dim, 1)

    def forward(self, x, query_embeddings=None, query_mask=None):
        """Computes class prediction logits.

        Query embeddings from a text encoder define the classification label space.

        Args:
        x: Image features [batch_size, num_patches, emb_dim].
        query_embeddings: The embeddings to classify against of shape [batch_size,
            num_queries, out_dim]. If not specified, only the image class embeddings
            will be returned.
        query_mask: Mask indicating whether query is real (1) or padding (0), of
            shape [batch_size, num_queries].
        Returns:
        Dict with keys 'class_embeddings' and, if query embeddings were provided,
        'pred_logits'.
        """
        image_class_emb = self.proj(x)
        if query_embeddings is None:
            return {"class_embeddings": image_class_emb}

        if self.normalize:
            image_class_emb /= image_class_emb.norm(
                axis=-1, keepdims=True) + 1e-6
            query_embeddings /= query_embeddings.norm(
                axis=-1, keepdims=True) + 1e-6

        pred_logits = paddle.matmul(
            x=image_class_emb, y=query_embeddings, transpose_y=True)

        logit_shift = self.logit_shift(x)
        logit_scale = F.elu(self.logit_scale(x)) + 1
        pred_logits = (logit_shift + pred_logits) * logit_scale

        if query_mask is not None:
            if len(query_mask.shape) > 1:
                query_mask = query_mask.unsqueeze(-2)
            pred_logits = paddle.where(query_mask == 0, -1e6, pred_logits)

        return pred_logits, image_class_emb


@register
class OWLViTHead(nn.Layer):

    __inject__ = ['class_head, bbox_head', 'loss']

    def __init__(self, class_head, bbox_head, loss, box_bias='both'):
        super().__init__()

        self.class_head = class_head
        self.bbox_head = bbox_head
        self.box_bias = box_bias
        self.matcher = matcher
        self.loss = loss

    def box_predictor(self, image_features, feature_map):
        """Predicts bounding boxes from image features.

        Args:
        image_features: Feature tokens extracted from the image, returned by the
            `embedder` function.
        feature_map: A spatial re-arrangement of image_features, also returned by
            the `embedder` function.

        Returns:
        List of predicted boxes (cxcywh normalized to 0, 1) nested within
            a dictionary.
        """
        # Bounding box detection head [b, num_patches, 4].
        pred_boxes = self.obj_box_head(image_features)
        # We compute the location of each token on the grid and use it to compute
        # a bias for the bbox prediction, i.e., each token is biased towards
        # predicting its location on the grid as the center.
        pred_boxes += compute_box_bias(feature_map, kind=self.box_bias)
        pred_boxes = nn.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(self,
                        image_features,
                        query_embeddings=None,
                        query_mask=None):
        """Applies the class head to the image features.

        Args:
        image_features: Feature tokens extracted by the image embedder.
        query_embeddings: Optional list of text (or image) embeddings. If no
            embeddings are provided, no logits will be computed and only the class
            embeddings for the image will be returned.
        query_mask: Must be provided with query_embeddings. A mask indicating
            which query embeddings are valid.

        Returns:
        A dictionary containing the class_embeddings and the pred_logits if
            query_embeddings and query_mask are provided.
        """
        return self.class_head(image_features, query_embeddings, query_mask)

    def forward(self, feature_map, query_embeddings, targets=None):
        b, c, h, w = feature_map.shape
        image_features = paddle.reshape(feature_map, (b, c, h * w))
        pred_boxes = self.box_predictor(image_features, feature_map)

        query_mask = (text_queries[..., 0] > 0).cast(paddle.float32)
        pred_logits, image_class_emb = self.class_predictor(
            image_features, query_embeddings, query_mask)

        if self.training:
            return self.get_loss([pred_boxes, pred_logits], targets)
        else:
            return self.get_pred(pred_boxes, pred_logits)

    def get_loss(self, head_outs, gt_meta):
        return self.loss(head_outs, gt_meta)
