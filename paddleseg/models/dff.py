# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time

import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

import paddle
from matplotlib import pyplot as plt
from scipy.ndimage import correlate1d

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers, ResNet18_vd, ResNet50_vd


@manager.MODELS.add_component
class DFF(nn.Layer):
    r"""Dynamic Feature Fusion for Semantic Edge Detection

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Yuan Hu, Yunpeng Chen, Xiang Li, Jiashi Feng. "Dynamic Feature Fusion
        for Semantic Edge Detection" *IJCAI*, 2019

    """

    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2D, pretrained=None, **kwargs):
        super(DFF, self).__init__()
        self.nclass = nclass
        self.backbone = backbone

        self.ada_learner = LocationAdaptiveLearner(nclass, nclass * 4, nclass * 4, norm_layer=norm_layer)

        self.side1 = nn.Sequential(nn.Conv2D(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2D(64, 1, 1),
                                   norm_layer(1),
                                   nn.Conv2DTranspose(1, 1, 4, stride=2, padding=1))
        self.side3 = nn.Sequential(nn.Conv2D(128, 1, 1),
                                   norm_layer(1),
                                   nn.Conv2DTranspose(1, 1, 8, stride=4, padding=2))
        self.side5 = nn.Sequential(nn.Conv2D(512, nclass, 1),
                                   norm_layer(nclass),
                                   nn.Conv2DTranspose(nclass, nclass, 32, stride=16, padding=8))

        self.side5_w = nn.Sequential(nn.Conv2D(512, nclass * 4, 1),
                                     norm_layer(nclass * 4),
                                     nn.Conv2DTranspose(nclass * 4, nclass * 4, 32, stride=16, padding=8))
        self.backbone = backbone
        self.backbone_indices = [1, 2, 3, 4]
        fpn_dim = 128
        inplane_head = 512
        fpn_inplanes = [64, 128, 256, 512]

        self.head = SFNetHead(
            inplane=inplane_head,
            num_class=nclass,
            fpn_inplanes=fpn_inplanes,
            fpn_dim=fpn_dim)

        # self.transfer = nn.Sequential(nn.Conv2D(nclass, 2 * nclass, 1),
        #                               nn.Conv2D(2 * nclass, nclass, 3, padding=1, groups=nclass))

        if pretrained is not None:
            utils.load_entire_model(self, pretrained)

    def forward(self, x):
        # print('input x.shape', x.shape)
        c1, c2, c3, _, c5 = feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        seg_out = self.head(feats)  # [b,19,256,256]
        s = F.interpolate(
            seg_out,
            paddle.shape(x)[2:],
            mode='bilinear')  # [b,19,h,w]
        if not self.training:
            return [s]
        side1 = self.side1(c1)  # (N, 1, 512, 512)
        side2 = self.side2(c2)  # (N, 1, 512, 512)
        side3 = self.side3(c3)  # (N, 1, 512, 512)
        side5 = self.side5(c5)  # (N, 19, H, W)
        side5_w = self.side5_w(c5)  # (N, 19*4, H, W)

        ada_weights = self.ada_learner(side5_w)  # (N, 19, 4, H, W)

        slice5 = side5[:, 0:1, :, :]  # (N, 1, H, W)
        fuse = paddle.concat((slice5, side1, side2, side3), 1)
        for i in range(side5.shape[1] - 1):
            slice5 = side5[:, i + 1:i + 2, :, :]  # (N, 1, H, W)
            fuse = paddle.concat((fuse, slice5, side1, side2, side3), 1)  # (N, 19*4, H, W)

        fuse = paddle.reshape(fuse, [fuse.shape[0], self.nclass, -1, fuse.shape[2], fuse.shape[3]])  # (N, 19, 4, H, W)
        fuse = paddle.multiply(fuse, ada_weights)  # (N, 19, 4, H, W)
        fuse = paddle.sum(fuse, 2)  # (N, 19, H, W)

        sed_outputs = [side5, fuse]

        ss_mask = paddle.nn.functional.softmax(seg_out, axis=1)
        inferred_sed = paddle.abs(ss_mask - F.avg_pool2d(ss_mask,
                                                         kernel_size=3,
                                                         stride=1, padding=1))
        # argmax_seg = paddle.argmax(seg_out, axis=1)
        # oneshot_seg = paddle.nn.functional.one_hot(argmax_seg, seg_out.shape[1])
        # oneshot_seg = paddle.transpose(oneshot_seg, [0, 3, 1, 2])
        # gy = correlate1d(oneshot_seg.numpy(), weights=[-1, 1], axis=3)
        # gx = correlate1d(oneshot_seg.numpy(), weights=[-1, 1], axis=2)
        # seg_gradient = gx + gy
        # seg_gradient[seg_gradient != 0] = 1
        # seg_gradient = paddle.to_tensor(seg_gradient)

        sed_outputs = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear') for logit in sed_outputs
        ]

        sed_inferred_out = F.interpolate(
            inferred_sed,
            paddle.shape(x)[2:],
            mode='bilinear')  # [b,19,h,w]

        # visual([seg_out[:, 0], seg_gradient[:, 0], sed_inferred_out[:, 0], sed_outputs[0][:, 0], sed_outputs[1][:, 0]])
        # visual([paddle.to_tensor(seg_gradient)[:,0], paddle.nn.functional.sigmoid(sed_outputs[1])[:, 0]])
        # intersec = oneshot_seg * paddle.nn.functional.sigmoid(sed_outputs[1])
        # visual([intersec[:, 0], paddle.nn.functional.sigmoid(sed_outputs[1])[:, 0], oneshot_seg[:, 0]])
        return [s, *sed_outputs, paddle.nn.functional.sigmoid(sed_outputs[1]), paddle.nn.functional.sigmoid(sed_inferred_out)]

def visual(images):
    fig = plt.figure()
    total = len(images)
    columns = max(total // 2, 1)
    rows = int(np.ceil(total / columns))
    for i in range(total):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.imshow(postprocess(images[i]), cmap=plt.cm.gray_r)
    plt.savefig('output/dff-with-floodnet/{}_side.png'.format(time.time()))
    plt.show()


def postprocess(outputs):
    results = paddle.squeeze(outputs, [0, 1])
    results = paddle.nn.functional.sigmoid(results)
    results *= 255.0
    results = results.cast('uint8')
    return results.numpy()


class LocationAdaptiveLearner(nn.Layer):
    """docstring for LocationAdaptiveLearner"""

    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2D):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2D(in_channels, out_channels, 1),
                                   norm_layer(out_channels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2D(out_channels, out_channels, 1),
                                   norm_layer(out_channels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2D(out_channels, out_channels, 1),
                                   norm_layer(out_channels))

    def forward(self, x):
        # x:side5_w (N, 19*4, H, W)
        x = self.conv1(x)  # (N, 19*4, H, W)
        x = self.conv2(x)  # (N, 19*4, H, W)
        x = self.conv3(x)  # (N, 19*4, H, W)
        x = paddle.reshape(x, [x.shape[0], self.nclass, -1, x.shape[2], x.shape[3]])  # (N, 19, 4, H, W)
        return x


class SFNetHead(nn.Layer):
    """
    The SFNetHead implementation.

    Args:
        inplane (int): Input channels of PPM module.
        num_class (int): The unique number of target classes.
        fpn_inplanes (list): The feature channels from backbone.
        fpn_dim (int, optional): The input channels of FAM module. Default: 256.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
    """

    def __init__(self,
                 inplane,
                 num_class,
                 fpn_inplanes,
                 fpn_dim=256,
                 enable_auxiliary_loss=False):
        super(SFNetHead, self).__init__()
        self.ppm = layers.PPModule(
            in_channels=inplane,
            out_channels=fpn_dim,
            bin_sizes=(1, 2, 3, 6),
            dim_reduction=True,
            align_corners=True)
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.fpn_in = []

        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2D(fpn_inplane, fpn_dim, 1),
                    layers.SyncBatchNorm(fpn_dim), nn.ReLU()))

        self.fpn_in = nn.LayerList(self.fpn_in)
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            if self.enable_auxiliary_loss:
                self.dsn.append(
                    nn.Sequential(layers.AuxLayer(fpn_dim, fpn_dim, num_class)))
        if self.enable_auxiliary_loss:
            self.dsn = nn.LayerList(self.dsn)

        self.conv_last = nn.Sequential(
            layers.ConvBNReLU(
                len(fpn_inplanes) * fpn_dim, fpn_dim, 3, bias_attr=False),
            nn.Conv2D(fpn_dim, num_class, kernel_size=1))

    def forward(self, conv_out):  # conv_out [b,64,256,256][b,128,128,128][b,256,64,64][b,512,32,32]
        psp_out = self.ppm(conv_out[-1])  # [b,128,32,32]
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            # f = self.fpn_out_align[i]([conv_x, f])
            # f = conv_x + f
            f = conv_x
            fpn_feature_list.append(f)
            if self.enable_auxiliary_loss:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [b,128,256,256][b,128,128,128][b,128,64,64][b,128,32,32]
        output_size = paddle.shape(fpn_feature_list[0])[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                F.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode='bilinear',
                    align_corners=True))
        fusion_out = paddle.concat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        return x


if __name__ == '__main__':
    x = paddle.randn([4, 3, 1024, 1024])
    model = DFF(19, ResNet18_vd())
    model(x)
