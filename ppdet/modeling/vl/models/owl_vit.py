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
from ppdet.modeling.architectures import BaseArch
from ..utils import seq2img
from ..tokenizer import tokenize


@register
class OWLViT(BaseArch):
    __category__ = 'architecture'

    def __init__(self, embedder, head):
        super().__init__()
        self.backbone = embedder
        self.head = head

    def tokenize(self, text, max_token_len):
        return tokenize(text, max_token_len)

    def image_embedder(self, images):
        """Embeds images into feature maps.

        Args:
        images: images of shape (batch, input_size, input_size, 3), scaled to the
            input range defined in the config. Padding should be at the bottom right
            of the image.

        Returns:
        A 2D map of image features.
        """
        image_features, _ = self.backbone(images=images)
        return seq2img(images, image_features)

    def text_embedder(self, text_queries):
        """Embeds text into features.

        Args:
        text_queries: int32 tokenized text queries of shape [..., num_tokens].

        Returns:
        An array of the same shape as text_queries, except for the last dimension,
        which is num_dimensions instead of num_tokens.
        """
        _, text_features = self.backbone(texts=text_queries)
        return text_features

    def forward(self, inputs, text_queries):
        """Applies TextZeroShotDetectionModule on the input.

        Args:
        inputs: Images [batch_size, height, width, 3].
        text_queries: Queries to score boxes on. Queries starting with 0 stand for
            padding [batch_size=b, num_queries=q, max_query_length=l].

        Returns:
        Outputs dict with items:
            pred_logits: Class logits [b, num_patches, num_queries].
            pred_boxes: Predicted bounding boxes [b, num_patches, 4].
            feature_map: Image embeddings 2d feature map [b, sp, sp, img_emb_dim].
        """
        # Embed images:
        feature_map = self.image_embedder(inputs)
        # Embed queries:
        query_embeddings = self.text_embedder(text_queries)
        outputs = self.head(feature_map, query_embeddings)
        return outputs
