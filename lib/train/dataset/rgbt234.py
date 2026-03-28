import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ..data.image_loader import jpeg4py_loader
from ..admin.environment import env_settings


class RGBT234(BaseVideoDataset):
    """ GOTO dataset.
    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf
    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().rgbt234_dir if root is None else root
        if not root:
            raise RuntimeError('RGBT234 dataset path is not configured. Please set `rgbt234_dir` in lib/train/admin/local.py.')
        super().__init__('RGBT234', root, image_loader)


        self.sequence_list = self._get_sequence_list()


        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()
        self.seq_img_dict = self._get_sequence_dict()


    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class


    def get_name(self):
        return 'rgbt234'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):


        dir_list = os.listdir(self.root)
        dir_list.sort()
        return dir_list


    def _get_sequence_dict(self):
        seq_dict = {'visible':{},'infrared':{}}
        seq_list = os.listdir(self.root)
        for seq in seq_list:
            img_list = sorted(os.listdir(os.path.join(self.root,seq,'visible')))
            tir_list = sorted(os.listdir(os.path.join(self.root,seq,'infrared')))
            seq_dict['visible'][seq] = img_list
            seq_dict['infrared'][seq] = tir_list
        return seq_dict

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "visible.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        tbb_anno_file = os.path.join(seq_path, "infrared.txt")
        tgt = pandas.read_csv(tbb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt),torch.tensor(tgt)

    def _read_target_visible(self, seq_path):

        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        rgbbbox,tbbox = self._read_bb_anno(seq_path)

        valid = (rgbbbox[:, 2] > 0) & (rgbbbox[:, 3] > 0)
        visible = valid

        return {'bbox': rgbbbox, 'valid': valid, 'visible': visible, 'tbox':tbbox}


    def _get_frame_path(self, seq_path, frame_id):
        seq = seq_path.split('/')[-1]
        img_vis_path = os.path.join(seq_path, 'visible', '{}'.format(self.seq_img_dict['visible'][seq][frame_id]))
        try:
            img_tir_path = os.path.join(seq_path, 'infrared', '{}'.format(self.seq_img_dict['infrared'][seq][frame_id]))
            if not os.path.exists(img_tir_path):
                a = 1
            vis_id = int(self.seq_img_dict['visible'][seq][frame_id].split('.')[0][:-1])
            tir_id = int(self.seq_img_dict['infrared'][seq][frame_id].split('.')[0][:-1])
            if abs(vis_id - tir_id) <= 10:
                frame_list = [int(frame.split('.')[0][:-1]) for frame in
                              self.seq_img_dict['infrared'][seq][max(frame_id - 5, 0):frame_id + 5]]
                frame = min(frame_list, key=lambda x: abs(x - vis_id))
                frame_index_shift = frame_list.index(frame) - frame_id + max(frame_id - 5, 0)
                tir_frame_id = frame_id + frame_index_shift
                img_tir_path = os.path.join(seq_path, 'infrared',
                                            '{}'.format(self.seq_img_dict['infrared'][seq][tir_frame_id]))


        except:
            img_tir_path = img_vis_path


        return img_vis_path, img_tir_path

    def _get_rgb_frame(self, seq_path, frame_id):
        img_vis_path, img_tir_path = self._get_frame_path(seq_path,frame_id)
        assert os.path.exists(img_vis_path), 'rgb image {} not exist'.format(img_vis_path)

        image_vis = self.image_loader(img_vis_path)
        return image_vis

    def _get_tir_frame(self, seq_path, frame_id):
        img_vis_path, img_tir_path = self._get_frame_path(seq_path,frame_id)
        assert os.path.exists(img_vis_path), 't image {} not exist'.format(img_tir_path)

        image_tir = self.image_loader(img_tir_path)

        return image_tir

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        rgb_frame_list = [self._get_rgb_frame(seq_path, f_id) for f_id in frame_ids]
        tir_frame_list = [self._get_tir_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return rgb_frame_list,tir_frame_list, anno_frames, object_meta


    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-1]
        return raw_class
