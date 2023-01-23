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

"""Model function definition, including both architecture and loss."""

from __future__ import absolute_import

from TensorFlow.common.tb_utils import ExamplesPerSecondEstimatorHook
import tensorflow.compat.v1 as tf
from absl import logging
import functools

import anchors
import efficientdet_arch_tl
import efficientdet_arch
import hparams_config_tl
import utils
from horovod_estimator import show_model
try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None
from horovod_estimator.hooks import BroadcastGlobalVariablesHook, LoggingTensorHook
from det_model_fn import update_learning_rate_schedule_parameters, learning_rate_schedule, \
                        detection_loss, reg_l2_loss, coco_metric_fn, add_metric_fn_inputs
from horovod_estimator.utis import is_rank0

def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model definition entry.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.

  Raises:
    RuntimeError: if both ckpt and backbone_ckpt are set.
  """
  # Convert params (dict) to Config for easier access.
  def _model_outputs():
    return model(features, config=hparams_config_tl.Config(params))

  if params['use_bfloat16']:
    with tf.tpu.bfloat16_scope():
      cls_outputs, box_outputs = _model_outputs()
      levels = cls_outputs.keys()
      for level in levels:
        cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = _model_outputs()
    levels = cls_outputs.keys()

  if is_rank0():
    show_model()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
    }
    for level in levels:
      predictions['cls_outputs_%d' % level] = cls_outputs[level]
      predictions['box_outputs_%d' % level] = box_outputs[level]
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Set up training loss and learning rate.
  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(params, global_step)

  # cls_loss and box_loss are for logging. only total_loss is optimized.
  det_loss, cls_loss, box_loss = detection_loss(cls_outputs, box_outputs,
                                                labels, params)
  l2loss = reg_l2_loss(params['weight_decay'])
  total_loss = det_loss + l2loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    utils.scalar('lrn_rate', learning_rate)
    utils.scalar('trainloss/cls_loss', cls_loss)
    utils.scalar('trainloss/box_loss', box_loss)
    utils.scalar('trainloss/det_loss', det_loss)
    utils.scalar('trainloss/l2_loss', l2loss)
    utils.scalar('trainloss/loss', total_loss)
    utils.scalar('loss', total_loss)  # for consistency

  moving_average_decay = params['moving_average_decay']
  if moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=moving_average_decay, num_updates=global_step)
    ema_vars = utils.get_ema_vars()

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])

    if hvd is not None and hvd.is_initialized():
        optimizer = hvd.DistributedOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list = tf.trainable_variables()
    if variable_filter_fn:
      var_list = variable_filter_fn(var_list)#, params['resnet_depth'])

    if params.get('clip_gradients_norm', 0) > 0:
      logging.info('clip gradients norm by %f', params['clip_gradients_norm'])
      grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
      with tf.name_scope('clip'):
        grads = [gv[0] for gv in grads_and_vars]
        tvars = [gv[1] for gv in grads_and_vars]
        clipped_grads, gnorm = tf.clip_by_global_norm(
            grads, params['clip_gradients_norm'])
        utils.scalar('gnorm', gnorm)
        grads_and_vars = list(zip(clipped_grads, tvars))

      with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    else:
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            total_loss, global_step, var_list=var_list)

    if moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(**kwargs):
      """Returns a dictionary that has the evaluation metrics."""
      batch_size = params['batch_size']
      eval_anchors = anchors.Anchors(params['min_level'],
                                     params['max_level'],
                                     params['num_scales'],
                                     params['aspect_ratios'],
                                     params['anchor_scale'],
                                     params['image_size'])
      anchor_labeler = anchors.AnchorLabeler(eval_anchors,
                                             params['num_classes'])
      cls_loss = tf.metrics.mean(kwargs['cls_loss_repeat'])
      box_loss = tf.metrics.mean(kwargs['box_loss_repeat'])

      if params.get('testdev_dir', None):
        logging.info('Eval testdev_dir %s', params['testdev_dir'])
        coco_metrics = coco_metric_fn(
            batch_size,
            anchor_labeler,
            params['val_json_file'],
            testdev_dir=params['testdev_dir'],
            disable_pyfun=params.get('disable_pyfun', None),
            **kwargs)
      else:
        logging.info('Eval val with groudtruths %s.', params['val_json_file'])
        coco_metrics = coco_metric_fn(batch_size, anchor_labeler,
                                      params['val_json_file'], **kwargs)

      # Add metrics to output.
      output_metrics = {
          'cls_loss': cls_loss,
          'box_loss': box_loss,
      }
      output_metrics.update(coco_metrics)
      return output_metrics

    cls_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(cls_loss, 0), [params['batch_size'],]),
        [params['batch_size'], 1])
    box_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(box_loss, 0), [params['batch_size'],]),
        [params['batch_size'], 1])
    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'box_loss_repeat': box_loss_repeat,
        'source_ids': labels['source_ids'],
        'groundtruth_data': labels['groundtruth_data'],
        'image_scales': labels['image_scales'],
    }
    add_metric_fn_inputs(params, cls_outputs, box_outputs, metric_fn_inputs)
    eval_metrics = (metric_fn, metric_fn_inputs)

  checkpoint = params.get('ckpt') or params.get('backbone_ckpt')
  if checkpoint and mode == tf.estimator.ModeKeys.TRAIN:
    # Initialize the model from an EfficientDet or backbone checkpoint.
    if params.get('ckpt') and params.get('backbone_ckpt'):
      raise RuntimeError(
        '--backbone_ckpt and --checkpoint are mutually exclusive')
    elif params.get('backbone_ckpt'):
      var_scope = params['backbone_name'] + '/'
      if params['ckpt_var_scope'] is None:
        # Use backbone name as default checkpoint scope.
        ckpt_scope = params['backbone_name'] + '/'
      else:
        ckpt_scope = params['ckpt_var_scope'] + '/'
    else:
      # Load every var in the given checkpoint
      var_scope = ckpt_scope = '/'

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      logging.info('restore variables from %s', checkpoint)

      var_map = utils.get_ckpt_var_map(
        ckpt_path=checkpoint,
        ckpt_scope=ckpt_scope,
        var_scope=var_scope,
        var_exclude_expr=params.get('var_exclude_expr', None))

      tf.train.init_from_checkpoint(checkpoint, var_map)

      return tf.train.Scaffold()
  elif mode == tf.estimator.ModeKeys.EVAL and moving_average_decay:
    def scaffold_fn():
      """Load moving average variables for eval."""
      logging.info('Load EMA vars with ema_decay=%f', moving_average_decay)
      restore_vars_dict = ema.variables_to_restore(ema_vars)
      saver = tf.train.Saver(restore_vars_dict)
      return tf.train.Scaffold(saver=saver)
  else:
    scaffold_fn = None

  training_hooks = []
  if hvd is not None and hvd.is_initialized():
    init_weights_hook = BroadcastGlobalVariablesHook(root_rank=0, model_dir=params['model_dir'])
    training_hooks.append(init_weights_hook)

  if is_rank0() or params['dump_all_ranks']:
    training_hooks.extend([
      LoggingTensorHook(
        dict(utils.summaries), summary_dir=params['model_dir'],
        every_n_iter=params['every_n_iter']),
      ExamplesPerSecondEstimatorHook(
        params['batch_size'], every_n_steps=params['every_n_iter'],
        output_dir=params['model_dir'], log_global_step=True)
    ])

  if mode == tf.estimator.ModeKeys.TRAIN:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      scaffold=scaffold_fn() if scaffold_fn is not None else None,
      training_hooks=training_hooks
    )
  else:
    # host_call in the original code was to write summary, but caused the error
    # 'ValueError: Tensor("strided_slice_6:0", shape=(), dtype=int64) must be from the same graph as Tensor("strided_slice:0", shape=(), dtype=float32)'
    # thus, it's handled in write_summary() in main.py
    return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metrics=eval_metrics,
      #host_call=utils.get_tpu_host_call(global_step, params),
      scaffold_fn=scaffold_fn)

def efficientdet_model_fn(features, labels, mode, params):
  """EfficientDet model."""
  variable_filter_fn = functools.partial(
      efficientdet_arch_tl.freeze_vars, pattern=params['var_freeze_expr'])
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=efficientdet_arch.efficientdet,
      variable_filter_fn=variable_filter_fn)

def get_model_fn(model_name='efficientdet-d0'):
  """Get model fn for a given model name."""
  if 'efficientdet' in model_name:
    return efficientdet_model_fn
  else:
    raise ValueError('Invalide model name {}'.format(model_name))
