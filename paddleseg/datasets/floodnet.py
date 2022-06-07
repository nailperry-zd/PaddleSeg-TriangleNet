# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import glob

import numpy as np
import paddle
import scipy.io
from PIL import Image

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F
import paddle.nn.functional as F


@manager.DATASETS.add_component
class FloodNet(Dataset):
    """
    FloodNet dataset `https://drive.google.com/drive/folders/1g1r419bWBe4GEF-7si5DqWCjxiC8ErnY?usp=sharing`.
    The folder structure is as follow:

        floodnet
        |
        |  |--train
        |  |--val
        |  |--test


    Args:
        transforms (list): Transforms for image.
        dataset_root (str): dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 10

    def __init__(self, transforms, dataset_root, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = mode + '-org-img'
        label_dir = mode + '-label-img'
        # if self.dataset_root is None or not os.path.isdir(
        #         self.dataset_root) or not os.path.isdir(
        #             img_dir) or not os.path.isdir(label_dir):
        #     raise ValueError(
        #         "The dataset is not Found or the folder structure is nonconfoumance."
        #     )

        label_files = sorted(
            glob.glob(
                os.path.join(self.dataset_root, mode, label_dir, '*_lab.png')))
        img_files = sorted(
            glob.glob(os.path.join(self.dataset_root, mode, img_dir, '*.jpg')))

        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'train':
            label_ss = np.asarray(Image.open(label_path))
            im, label = self.transforms(im=image_path, label=label_ss)
            # 从semantic segmentation label中提取semantic edge, 使用adaptive pool
            label_tensor = paddle.to_tensor(label).astype('int64')
            # 去除255
            label_valid_mask = label_tensor != self.ignore_index
            label_valid = label_tensor * label_valid_mask
            oneshot_seg = paddle.nn.functional.one_hot(label_valid, self.num_classes)
            oneshot_seg = paddle.transpose(oneshot_seg, [2, 0, 1]).unsqueeze(0)
            inferred_sed_gt = paddle.abs(oneshot_seg - F.avg_pool2d(oneshot_seg,
                                                             kernel_size=3,
                                                             stride=1, padding=1))
            inferred_sed_gt[inferred_sed_gt > 0] = 1
            inferred_sed_gt = inferred_sed_gt.squeeze(0).astype('int64')
            inferred_sed_gt_list = [inferred_sed_gt[i, :, :].unsqueeze(0) for i in range(self.num_classes)]
            inferred_sed_gt_list.append(paddle.to_tensor(label).unsqueeze(0).astype('int64'))
            label = paddle.stack(inferred_sed_gt_list, axis=1)  # [c+1, h, w]
            return im, label.squeeze(0).numpy()
        else:
            label_ss = np.asarray(Image.open(label_path))
            # label也要同步变换，比如图片翻转了，label也要翻转
            im, label = self.transforms(im=image_path, label=label_ss)
            return im, label