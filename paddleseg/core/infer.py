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

import collections.abc
import time
from itertools import combinations

import numpy as np
import cv2
import paddle
import paddle.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt


def get_reverse_list(ori_shape, transforms):
    """
    get reverse list of transform.

    Args:
        ori_shape (list): Origin shape of image.
        transforms (list): List of transform.

    Returns:
        list: List of tuple, there are two format:
            ('resize', (h, w)) The image shape before resize,
            ('padding', (h, w)) The image shape before padding.
    """
    reverse_list = []
    h, w = ori_shape[0], ori_shape[1]
    for op in transforms:
        if op.__class__.__name__ in ['Resize']:
            reverse_list.append(('resize', (h, w)))
            h, w = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['ResizeByLong']:
            reverse_list.append(('resize', (h, w)))
            long_edge = max(h, w)
            short_edge = min(h, w)
            short_edge = int(round(short_edge * op.long_size / long_edge))
            long_edge = op.long_size
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
        if op.__class__.__name__ in ['Padding']:
            reverse_list.append(('padding', (h, w)))
            w, h = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['PaddingByAspectRatio']:
            reverse_list.append(('padding', (h, w)))
            ratio = w / h
            if ratio == op.aspect_ratio:
                pass
            elif ratio > op.aspect_ratio:
                h = int(w / op.aspect_ratio)
            else:
                w = int(h * op.aspect_ratio)
        if op.__class__.__name__ in ['LimitLong']:
            long_edge = max(h, w)
            short_edge = min(h, w)
            if ((op.max_long is not None) and (long_edge > op.max_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.max_long
                short_edge = int(round(short_edge * op.max_long / long_edge))
            elif ((op.min_long is not None) and (long_edge < op.min_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.min_long
                short_edge = int(round(short_edge * op.min_long / long_edge))
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
    return reverse_list


def reverse_transform(pred, ori_shape, transforms, mode='nearest'):
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            if paddle.get_device() == 'cpu':
                pred = paddle.cast(pred, 'uint8')
                pred = F.interpolate(pred, (h, w), mode=mode)
                pred = paddle.cast(pred, 'int32')
            else:
                pred = F.interpolate(pred, (h, w), mode=mode)
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def flip_combination(flip_horizontal=False, flip_vertical=False):
    """
    Get flip combination.

    Args:
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.

    Returns:
        list: List of tuple. The first element of tuple is whether to flip horizontally,
            and the second is whether to flip vertically.
    """

    flip_comb = [(False, False)]
    if flip_horizontal:
        flip_comb.append((True, False))
    if flip_vertical:
        flip_comb.append((False, True))
        if flip_horizontal:
            flip_comb.append((True, True))
    return flip_comb


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def slide_inference(model, im, crop_size, stride):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = np.int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = np.int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = None
    count = np.zeros([1, 1, h_im, w_im])
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logits = model(im_crop)
            if not isinstance(logits, collections.abc.Sequence):
                raise TypeError(
                    "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                    .format(type(logits)))
            logit = logits[0].numpy()
            if final_logit is None:
                final_logit = np.zeros([1, logit.shape[1], h_im, w_im])
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    final_logit = paddle.to_tensor(final_logit)
    return final_logit


def inference(model,
              im,
              ori_shape=None,
              transforms=None,
              is_slide=False,
              stride=None,
              crop_size=None):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        im = im.transpose((0, 2, 3, 1))
    if not is_slide:
        logits = model(im)
        if not isinstance(logits, collections.abc.Sequence):
            raise TypeError(
                "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(logits)))
        # visual([logits[0][:,0], logits[0][:,1], logits[0][:,7], logits[0][:,13]])
        logit = logits[0]
    else:
        logit = slide_inference(model, im, crop_size=crop_size, stride=stride)
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        logit = logit.transpose((0, 3, 1, 2))
    if ori_shape is not None:
        pred = reverse_transform(logit, ori_shape, transforms, mode='bilinear')
        pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')
        return pred
    else:
        return logit


def visual(images):
    fig = plt.figure()
    total = len(images)
    columns = max(total // 2, 1)
    rows = int(np.ceil(total / columns))
    for i in range(total):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.imshow(postprocess(images[i]), cmap=plt.cm.gray_r)
    plt.savefig('output/dff/{}_label.png'.format(time.time()))
    plt.show()


def postprocess2(outputs):
    results = paddle.clip(outputs, 0, 1)
    results = paddle.squeeze(results, [0, 1])
    results *= 255.0
    results = results.cast('uint8')
    return results.numpy()


def save_nparray_as_img(img_path, nparr):
    im = Image.fromarray(postprocess(nparr))
    im.convert('L').save(img_path, format='jpeg')
    # fig = plt.figure()
    # plt.imshow(postprocess(nparr))
    # plt.savefig(img_path)
    # plt.show()


def postprocess(outputs):
    results = paddle.squeeze(outputs, [0, 1])
    results = paddle.nn.functional.sigmoid(results)
    results *= 255.0
    results = results.cast('uint8')
    return results.numpy()


def NormMinandMax(output, min=0, max=1):
    """"
    将数据 归一化到[min,max]区间的方法
    返回 副本
    """
    Ymax = paddle.max(output).numpy()[0]  # 计算最大值
    Ymin = paddle.min(output).numpy()[0]  # 计算最小值
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (output - Ymin)

    return last


# color map for each trainId, from the official cityscapes script
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
def get_colors():
    color_dict = {}
    color_dict[0] = [128, 64, 128]
    color_dict[1] = [244, 35, 232]
    color_dict[2] = [70, 70, 70]
    color_dict[3] = [102, 102, 156]
    color_dict[4] = [190, 153, 153]
    color_dict[5] = [153, 153, 153]
    color_dict[6] = [250, 170, 30]
    color_dict[7] = [220, 220, 0]
    color_dict[8] = [107, 142, 35]
    color_dict[9] = [152, 251, 152]
    color_dict[10] = [70, 130, 180]
    color_dict[11] = [220, 20, 60]
    color_dict[12] = [255, 0, 0]
    color_dict[13] = [0, 0, 142]
    color_dict[14] = [0, 0, 70]
    color_dict[15] = [0, 60, 100]
    color_dict[16] = [0, 80, 100]
    color_dict[17] = [0, 0, 230]
    color_dict[18] = [119, 11, 32]
    return color_dict


def visual_by_channel(score_output):
    _, num_cls, h, w = score_output.shape
    color_dict = get_colors()
    for idx_cls in range(num_cls):
        r = np.zeros((h, w))
        g = np.zeros((h, w))
        b = np.zeros((h, w))
        rgb = np.zeros((h, w, 3))
        score_pred = score_output[0, idx_cls].numpy()
        score_pred_flag = (score_pred > 0.3)  # 控制某个像素为边缘的概率阈值
        r[score_pred_flag == 1] = color_dict[idx_cls][0]
        g[score_pred_flag == 1] = color_dict[idx_cls][1]
        b[score_pred_flag == 1] = color_dict[idx_cls][2]
        r[score_pred_flag == 0] = 255
        g[score_pred_flag == 0] = 255
        b[score_pred_flag == 0] = 255
        rgb[:, :, 0] = (r / 255.0)
        rgb[:, :, 1] = (g / 255.0)
        rgb[:, :, 2] = (b / 255.0)
        plt.imsave("output/sfnet-origin/{}_feature_channel_{}.png".format(time.time(), idx_cls), rgb)

def visual_by_argmax(argmax):
    _, _, h, w = argmax.shape
    num_cls = 19
    color_dict = get_colors()
    one_hot = F.one_hot(argmax.squeeze(0).squeeze(0), num_cls)
    for idx_cls in range(num_cls):
        r = np.zeros((h, w))
        g = np.zeros((h, w))
        b = np.zeros((h, w))
        rgb = np.zeros((h, w, 3))
        score_pred = one_hot[:, :, idx_cls].numpy()
        score_pred_flag = (score_pred == 1)  # 控制某个像素为边缘的概率阈值
        r[score_pred_flag == 1] = color_dict[idx_cls][0]
        g[score_pred_flag == 1] = color_dict[idx_cls][1]
        b[score_pred_flag == 1] = color_dict[idx_cls][2]
        r[score_pred_flag == 0] = 255
        g[score_pred_flag == 0] = 255
        b[score_pred_flag == 0] = 255
        rgb[:, :, 0] = (r / 255.0)
        rgb[:, :, 1] = (g / 255.0)
        rgb[:, :, 2] = (b / 255.0)
        plt.imsave("output/sfnet-origin/{}_argmax_onehot_index_{}.png".format(time.time(), idx_cls), rgb)

def visual_all(score_output):
    _, num_cls, h, w = score_output.shape
    color_dict = get_colors()
    r = np.zeros((h, w))#默认黑色表示背景
    g = np.zeros((h, w))
    b = np.zeros((h, w))
    rgb = np.zeros((h, w, 3))
    multi_label_mask = np.zeros((h, w))
    for idx_cls in range(num_cls):
        score_pred = score_output[0, idx_cls].numpy()
        score_pred_flag = (score_pred > 0.3)  # 控制某个像素为边缘的概率阈值
        target_label_index = (score_pred_flag == 1)
        r[target_label_index] = color_dict[idx_cls][0]
        g[target_label_index] = color_dict[idx_cls][1]
        b[target_label_index] = color_dict[idx_cls][2]
        multi_label_mask[target_label_index] += 1
        # r[score_pred_flag == 0] = 255
        # g[score_pred_flag == 0] = 255
        # b[score_pred_flag == 0] = 255
    multi_label_index = (multi_label_mask > 1)
    # r[multi_label_index] = 255 #边界处用白色描边
    # g[multi_label_index] = 255
    # b[multi_label_index] = 255
    rgb[:, :, 0] = (r / 255.0)
    rgb[:, :, 1] = (g / 255.0)
    rgb[:, :, 2] = (b / 255.0)
    plt.imsave("output/sfnet-origin/visual_all_classes_{}.png".format(time.time()), rgb)

def aug_inference(model,
                  im,
                  ori_shape,
                  transforms,
                  scales=1.0,
                  flip_horizontal=False,
                  flip_vertical=False,
                  is_slide=False,
                  stride=None,
                  crop_size=None):
    """
    Infer with augmentation.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        scales (float|tuple|list):  Scales for resize. Default: 1.
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.
        is_slide (bool): Whether to infer by sliding wimdow. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: Prediction of image with shape (1, 1, h, w) is returned.
    """
    if isinstance(scales, float):
        scales = [scales]
    elif not isinstance(scales, (tuple, list)):
        raise TypeError(
            '`scales` expects float/tuple/list type, but received {}'.format(
                type(scales)))
    final_logit = 0
    h_input, w_input = im.shape[-2], im.shape[-1]
    flip_comb = flip_combination(flip_horizontal, flip_vertical)
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im = F.interpolate(im, (h, w), mode='bilinear')
        for flip in flip_comb:
            im_flip = tensor_flip(im, flip)
            logit = inference(
                model,
                im_flip,
                is_slide=is_slide,
                crop_size=crop_size,
                stride=stride)
            logit = tensor_flip(logit, flip)
            logit = F.interpolate(logit, (h_input, w_input), mode='bilinear')

            logit = F.softmax(logit, axis=1)
            final_logit = final_logit + logit

    pred = reverse_transform(
        final_logit, ori_shape, transforms, mode='bilinear')
    pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')
    return pred
