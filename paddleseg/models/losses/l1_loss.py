# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
from paddle import nn, fluid

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class L1Loss(nn.MSELoss):
    r"""
    This interface is used to construct a callable object of the ``L1Loss`` class.
    The L1Loss layer calculates the L1 Loss of ``input`` and ``label`` as follows.
     If `reduction` set to ``'none'``, the loss is:
    .. math::
        Out = \lvert input - label\rvert
    If `reduction` set to ``'mean'``, the loss is:
    .. math::
        Out = MEAN(\lvert input - label\rvert)
    If `reduction` set to ``'sum'``, the loss is:
    .. math::
        Out = SUM(\lvert input - label\rvert)

    Args:
        reduction (str, optional): Indicate the reduction to apply to the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned;
            If `reduction` is ``'mean'``, the reduced mean loss is returned.
            If `reduction` is ``'sum'``, the reduced sum loss is returned.
            Default is ``'mean'``.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: 255.
    Shape:
        input (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        label (Tensor): label. The shapes is [N, *], same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        output (Tensor): The L1 Loss of ``input`` and ``label``.
            If `reduction` is ``'none'``, the shape of output loss is [N, *], the same as ``input`` .
            If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1].
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            input_data = np.array([[1.5, 0.8], [0.2, 1.3]]).astype("float32")
            label_data = np.array([[1.7, 1], [0.4, 0.5]]).astype("float32")
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)
            l1_loss = paddle.nn.L1Loss()
            output = l1_loss(input, label)
            print(output.numpy())
            # [0.35]
            l1_loss = paddle.nn.L1Loss(reduction='sum')
            output = l1_loss(input, label)
            print(output.numpy())
            # [1.4]
            l1_loss = paddle.nn.L1Loss(reduction='none')
            output = l1_loss(input, label)
            print(output)
            # [[0.20000005 0.19999999]
            # [0.2        0.79999995]]
    """

    def __init__(self, reduction='mean', ignore_index=255, weight='dynamic'):
        self.ignore_index = ignore_index
        self.EPS = 1e-10
        self.weight = weight
        if self.weight is not None:
            if isinstance(self.weight, str):
                if self.weight != 'dynamic':
                    raise ValueError(
                        "if type of `weight` is str, it should equal to 'dynamic', but it is {}"
                        .format(self.weight))
            elif isinstance(self.weight, paddle.VarBase):
                raise TypeError(
                    'The type of `weight` is wrong, it should be Tensor or str, but it is {}'
                    .format(type(self.weight)))

        super().__init__(reduction=reduction)

    def forward(self, input, label):
        mask = (label != self.ignore_index)#裁剪填充不参与计算
        mask = paddle.cast(mask, 'float32')

        if isinstance(self.weight, str):
            pos_index = (label == 1)
            neg_index = (label == 0)
            pos_num = paddle.sum(pos_index.astype('float32'))
            neg_num = paddle.sum(neg_index.astype('float32'))
            sum_num = pos_num + neg_num
            weight_pos = 2 * neg_num / (sum_num + self.EPS)
            weight_neg = 2 * pos_num / (sum_num + self.EPS)
            weight = weight_pos * label + weight_neg * (1 - label)
        else:
            weight = self.weight

        if not fluid.framework.in_dygraph_mode():
            fluid.data_feeder.check_variable_and_dtype(
                input, 'input', ['float32', 'float64'], 'MSELoss')
            fluid.data_feeder.check_variable_and_dtype(
                label, 'label', ['float32', 'float64'], 'MSELoss')

        square_out = fluid.layers.square(
            fluid.layers.elementwise_sub(input, label.astype('float32')))
        square_out = square_out * mask# 去除裁剪填充项，也就是ignore
        if weight is not None:
            square_out = weight * square_out
        if self.reduction == 'none':
            return square_out

        reduce_op = 'reduce_mean'
        if self.reduction == 'sum':
            reduce_op = 'reduce_sum'

        mask.stop_gradient = True
        label.stop_gradient = True
        return getattr(fluid.layers, reduce_op)(square_out)
