# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import re
import shutil
import tempfile

import paddle
from ppcls.utils import logger
from .download import get_weights_path_from_url

__all__ = ['init_model', 'save_model', 'load_dygraph_pretrain']


def _mkdir_if_not_exist(path):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(f'be happy if some process has already created {path}')
            else:
                raise OSError(f'Failed to mkdir {path}')


def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(f'{path}.pdparams')):
        raise ValueError(f"Model pretrain path {path} does not exists.")
    param_state_dict = paddle.load(f"{path}.pdparams")
    model.set_dict(param_state_dict)
    return


def load_dygraph_pretrain_from_url(model, pretrained_url, use_ssld):
    if use_ssld:
        pretrained_url = pretrained_url.replace("_pretrained",
                                                "_ssld_pretrained")
    local_weight_path = get_weights_path_from_url(pretrained_url).replace(
        ".pdparams", "")
    load_dygraph_pretrain(model, path=local_weight_path)
    return


def load_distillation_model(model, pretrained_model):
    logger.info("In distillation mode, teacher model will be "
                "loaded firstly before student model.")

    if not isinstance(pretrained_model, list):
        pretrained_model = [pretrained_model]

    teacher = model.teacher if hasattr(model,
                                       "teacher") else model._layers.teacher
    student = model.student if hasattr(model,
                                       "student") else model._layers.student
    load_dygraph_pretrain(teacher, path=pretrained_model[0])
    logger.info(f"Finish initing teacher model from {pretrained_model}")
    # load student model
    if len(pretrained_model) >= 2:
        load_dygraph_pretrain(student, path=pretrained_model[1])
        logger.info(f"Finish initing student model from {pretrained_model}")


def init_model(config, net, optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    checkpoints = config.get('checkpoints')
    if checkpoints and optimizer is not None:
        assert os.path.exists(
            f"{checkpoints}.pdparams"
        ), f"Given dir {checkpoints}.pdparams not exist."
        assert os.path.exists(
            f"{checkpoints}.pdopt"
        ), f"Given dir {checkpoints}.pdopt not exist."
        para_dict = paddle.load(f"{checkpoints}.pdparams")
        opti_dict = paddle.load(f"{checkpoints}.pdopt")
        metric_dict = paddle.load(f"{checkpoints}.pdstates")
        net.set_dict(para_dict)
        optimizer.set_state_dict(opti_dict)
        logger.info(f"Finish load checkpoints from {checkpoints}")
        return metric_dict

    pretrained_model = config.get('pretrained_model')
    use_distillation = config.get('use_distillation', False)
    if pretrained_model:
        if use_distillation:
            load_distillation_model(net, pretrained_model)
        else:  # common load
            load_dygraph_pretrain(net, path=pretrained_model)
            logger.info(
                logger.coloring(
                    f"Finish load pretrained model from {pretrained_model}",
                    "HEADER",
                )
            )


def save_model(net,
               optimizer,
               metric_info,
               model_path,
               model_name="",
               prefix='ppcls'):
    """
    save model to the target path
    """
    if paddle.distributed.get_rank() != 0:
        return
    model_path = os.path.join(model_path, model_name)
    _mkdir_if_not_exist(model_path)
    model_path = os.path.join(model_path, prefix)

    paddle.save(net.state_dict(), f"{model_path}.pdparams")
    paddle.save(optimizer.state_dict(), f"{model_path}.pdopt")
    paddle.save(metric_info, f"{model_path}.pdstates")
    logger.info(f"Already save model in {model_path}")
