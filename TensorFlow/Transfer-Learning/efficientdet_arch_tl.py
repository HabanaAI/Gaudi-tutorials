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
"""EfficientDet model definition.

[1] Mingxing Tan, Ruoming Pang, Quoc Le.
    EfficientDet: Scalable and Efficient Object Detection.
    CVPR 2020, https://arxiv.org/abs/1911.09070
"""

from __future__ import absolute_import

import re
from absl import logging

def freeze_vars(variables, pattern):
  """Removes backbone+fpn variables from the input.
  Args:
    variables: all the variables in training
    pattern: a reg experession such as ".*(efficientnet|fpn_cells).*".
  Returns:
    var_list: a list containing variables for training
  """
  if pattern:
    filtered_vars = [v for v in variables if not re.match(pattern, v.name)]
    if len(filtered_vars) == len(variables):
      logging.warning('%s didnt match with any variable. Please use compatible '
                      'pattern. i.e "(efficientnet)"', pattern)
    return filtered_vars
  return variables
