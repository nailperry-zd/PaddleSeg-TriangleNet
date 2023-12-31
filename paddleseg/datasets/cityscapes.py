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
import scipy.io
from PIL import Image

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class Cityscapes(Dataset):
    """
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
    The folder structure is as follow:

        cityscapes
        |
        |--leftImg8bit
        |  |--train
        |  |--val
        |  |--test
        |
        |--gtFine
        |  |--train
        |  |--val
        |  |--test

    Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 19

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

        img_dir = os.path.join(self.dataset_root, 'leftImg8bit')
        label_dir = os.path.join(self.dataset_root, 'gtFine')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
            img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        label_files = sorted(
            glob.glob(
                os.path.join(label_dir, mode, '*',
                             '*_gtFine_labelTrainIds.png')))
        img_files = sorted(
            glob.glob(os.path.join(img_dir, mode, '*', '*_leftImg8bit.png')))

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
            file_name = os.path.basename(image_path)
            label_ss = np.asarray(Image.open(label_path))
            mask_path = os.path.join(self.dataset_root,
                                     'gt_eval/gt_raw/cls/{}'.format(file_name))
            mask_path = mask_path.replace('.png', '.mat')
            mat = scipy.io.loadmat(mask_path, mat_dtype=True, squeeze_me=True,
                                   struct_as_record=False)
            label = []
            for i in range(self.num_classes):
                label_i = mat['GTcls'].Boundaries[i].toarray()
                label.append(label_i)
            label.append(label_ss)
            # label也要同步变换，比如图片翻转了，label也要翻转
            im, label = self.transforms(im=image_path, label=label)
            label = np.stack(label, axis=0)# [1+19, h, w]
            return im, label
        else:
            file_name = os.path.basename(image_path)
            label_ss = np.asarray(Image.open(label_path))
            label = [label_ss]
            # label也要同步变换，比如图片翻转了，label也要翻转
            im, label = self.transforms(im=image_path, label=label)
            label = np.stack(label, axis=0)  # [1+19, h, w]
            return im, label