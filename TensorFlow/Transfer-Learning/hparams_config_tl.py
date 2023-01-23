###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
"""Hparams for model architecture and trainer."""

from __future__ import absolute_import

from hparams_config import Config
import hparams_config

def default_detection_configs():
  """Returns a default detection configs."""
  h = Config()

  # model name.
  h.name = 'efficientdet-d1'

  # input preprocessing parameters
  h.image_size = 640
  h.input_rand_hflip = True
  h.train_scale_min = 0.1
  h.train_scale_max = 2.0
  h.autoaugment_policy = None

  # dataset specific parameters
  h.num_classes = 90
  h.skip_crowd_during_training = True

  # model architecture
  h.min_level = 3
  h.max_level = 7
  h.num_scales = 3
  h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
  h.anchor_scale = 4.0
  # is batchnorm training mode
  h.is_training_bn = True
  # optimization
  h.momentum = 0.9
  h.learning_rate = 0.08
  h.lr_warmup_init = 0.008
  h.lr_warmup_epoch = 1.0
  h.first_lr_drop_epoch = 200.0
  h.second_lr_drop_epoch = 250.0
  h.clip_gradients_norm = 10.0
  h.num_epochs = 300

  # classification loss
  h.alpha = 0.25
  h.gamma = 1.5
  # localization loss
  h.delta = 0.1
  h.box_loss_weight = 50.0
  # regularization l2 loss.
  h.weight_decay = 4e-5
  # enable bfloat
  h.use_bfloat16 = True

  # For detection.
  h.box_class_repeats = 3
  h.fpn_cell_repeats = 3
  h.fpn_num_filters = 88
  h.separable_conv = True
  h.apply_bn_for_resampling = True
  h.conv_after_downsample = False
  h.conv_bn_relu_pattern = False
  h.use_native_resize_op = False
  h.pooling_type = None

  # version.
  h.fpn_name = None
  h.fpn_config = None

  # No stochastic depth in default.
  h.survival_prob = None

  h.lr_decay_method = 'cosine'
  h.moving_average_decay = 0.9998
  h.ckpt_var_scope = None  # ckpt variable scope.
  # exclude vars when loading pretrained ckpts.
  h.var_exclude_expr = '.*/class-predict/.*'  # exclude class weights in default

  h.backbone_name = 'efficientnet-b1'
  h.backbone_config = None
  h.var_freeze_expr = None

  # RetinaNet.
  h.resnet_depth = 50
  return h

def get_efficientdet_config(model_name='efficientdet-d1'):
  """Get the default config for EfficientDet based on model name."""
  h = default_detection_configs()
  h.override(hparams_config.efficientdet_model_param_dict[model_name])
  return h

def get_detection_config(model_name) -> Config:
  if model_name.startswith('efficientdet'):
    return get_efficientdet_config(model_name)
  else:
    raise ValueError('model name must start with efficientdet or retinanet.')
