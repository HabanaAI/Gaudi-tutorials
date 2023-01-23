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


"""Data loader and processing.

This module is borrowed from TPU RetinaNet implementation:
https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/anchors.py
"""

import tensorflow.compat.v1 as tf

from dataloader import InputReader, DetectionInputProcessor
from dataloader import pad_to_fixed_size
import anchors
try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None
from object_detection import tf_example_decoder

MAX_NUM_INSTANCES = 100

class DetectionInputProcessorTL(DetectionInputProcessor):
  """Input processor for transfer learning of object detection."""
  def __init__(self, image, output_size, boxes=None, classes=None):
    DetectionInputProcessor.__init__(self, image, output_size, boxes, classes)

  def normalize_image(self):
    """Normalize the image to zero mean and unit variance."""
    # The image normalization is identical to Cloud TPU ResNet.
    mean_rgb= [0.485 * 255, 0.456 * 255, 0.406 * 255]
    stddev_rgb=[0.229 * 255, 0.224 * 255, 0.225 * 255]

    self._image = tf.cast(self._image, dtype=tf.float32)
    self._image -= tf.constant(mean_rgb, shape=(1, 1, 3), dtype=tf.float32)
    self._image /= tf.constant(stddev_rgb, shape=(1, 1, 3), dtype=tf.float32)
    return self._image

class InputReaderTL(InputReader):
  """Input reader for transfer learning dataset."""
  def __init__(self, file_pattern, is_training, params=None, use_fake_data=False, is_deterministic=False):
    super().__init__(file_pattern, is_training, params, use_fake_data, is_deterministic)

  def __call__(self, params=None):
    if params is None:
      params = self._params
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    params['image_size'])
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
    example_decoder = tf_example_decoder.TfExampleDecoder()

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        image: Image tensor that is preprocessed to have normalized value and
          fixed dimension [image_size, image_size, 3]
        cls_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors]. The height_l and width_l
          represent the dimension of class logits at l-th level.
        box_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        num_positives: Number of positive anchors in the image.
        source_id: Source image id. Default value -1 if the source id is empty
          in the groundtruth annotation.
        image_scale: Scale of the processed image to the original image.
        boxes: Groundtruth bounding box annotations. The box is represented in
          [y1, x1, y2, x2] format. The tensor is padded with -1 to the fixed
          dimension [self._max_num_instances, 4].
        is_crowds: Groundtruth annotations to indicate if an annotation
          represents a group of instances by value {0, 1}. The tensor is
          padded with 0 to the fixed dimension [self._max_num_instances].
        areas: Groundtruth areas annotations. The tensor is padded with -1
          to the fixed dimension [self._max_num_instances].
        classes: Groundtruth classes annotations. The tensor is padded with -1
          to the fixed dimension [self._max_num_instances].
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        source_id = data['source_id']
        image = data['image']
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        areas = data['groundtruth_area']
        is_crowds = data['groundtruth_is_crowd']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])

        if params['skip_crowd_during_training'] and self._is_training:
          indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
          classes = tf.gather_nd(classes, indices)
          boxes = tf.gather_nd(boxes, indices)

        # NOTE: The autoaugment method works best when used alongside the
        # standard horizontal flipping of images along with size jittering
        # and normalization.
        if params.get('autoaugment_policy', None) and self._is_training:
          from aug import autoaugment  # pylint: disable=g-import-not-at-top
          image, boxes = autoaugment.distort_image_with_autoaugment(
              image, boxes, params['autoaugment_policy'])

        input_processor = DetectionInputProcessorTL(
            image, params['image_size'], boxes, classes)
        input_processor.normalize_image()
        if self._is_training and params['input_rand_hflip']:
          input_processor.random_horizontal_flip()
        if self._is_training:
          input_processor.set_training_random_scale_factors(
              params['train_scale_min'], params['train_scale_max'])
        else:
          input_processor.set_scale_factors_to_output_size()
        image = input_processor.resize_and_crop_image()
        boxes, classes = input_processor.resize_and_crop_boxes()

        # Assign anchors.
        (cls_targets, box_targets,
         num_positives) = anchor_labeler.label_anchors(boxes, classes)

        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        # Pad groundtruth data for evaluation.
        image_scale = input_processor.image_scale_to_original
        boxes *= image_scale
        is_crowds = tf.cast(is_crowds, dtype=tf.float32)
        boxes = pad_to_fixed_size(boxes, -1, [self._max_num_instances, 4])
        is_crowds = pad_to_fixed_size(is_crowds, 0,
                                      [self._max_num_instances, 1])
        areas = pad_to_fixed_size(areas, -1, [self._max_num_instances, 1])
        classes = pad_to_fixed_size(classes, -1, [self._max_num_instances, 1])
        if params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)
        return (image, cls_targets, box_targets, num_positives, source_id,
                image_scale, boxes, is_crowds, areas, classes)

    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=self._is_training)

    if hvd is not None and hvd.is_initialized() and self._is_training: #multi card eval is not supported yet
      # 根据 GPU 数量做 shard 均分
      dataset = dataset.shard(hvd.size(), hvd.rank())

    if self._is_training:
      dataset = dataset.repeat()

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    cycle_length = 1 if self._is_deterministic else 32
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _prefetch_dataset, cycle_length=cycle_length, sloppy=self._is_training))
    if self._is_training:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.
    num_parallel_calls = 1 if self._is_deterministic else 64
    dataset = dataset.map(_dataset_parser, num_parallel_calls=num_parallel_calls)
    batch_size = params['batch_size']
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def _process_example(images, cls_targets, box_targets, num_positives,
                         source_ids, image_scales, boxes, is_crowds, areas,
                         classes):
      """Processes one batch of data."""
      labels = {}
      # Count num_positives in a batch.
      num_positives_batch = tf.reduce_mean(num_positives)
      labels['mean_num_positives'] = tf.reshape(
          tf.tile(tf.expand_dims(num_positives_batch, 0), [
              batch_size,
          ]), [batch_size, 1])

      for level in range(params['min_level'], params['max_level'] + 1):
        labels['cls_targets_%d' % level] = cls_targets[level]
        labels['box_targets_%d' % level] = box_targets[level]
      # Concatenate groundtruth annotations to a tensor.
      groundtruth_data = tf.concat([boxes, is_crowds, areas, classes], axis=2)
      labels['source_ids'] = source_ids
      labels['groundtruth_data'] = groundtruth_data
      labels['image_scales'] = image_scales
      return images, labels

    dataset = dataset.map(_process_example)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if self._use_fake_data:
      # Turn this dataset into a semi-fake dataset which always loop at the
      # first batch. This reduces variance in performance and is useful in
      # testing.
      dataset = dataset.take(1).cache().repeat()
    return dataset