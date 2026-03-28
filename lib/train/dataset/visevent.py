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


class Visevent(BaseVideoDataset):
    """VisEvent dataset."""

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        root = env_settings().visevent_dir if root is None else root
        if not root:
            raise RuntimeError('VisEvent dataset path is not configured. Please set `visevent_dir` in lib/train/admin/local.py.')

        super().__init__('visevent', root, image_loader)
        self.split = split
        self.sequence_list = self._get_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.seq_per_class = self._build_class_list()
        self.seq_img_dict = self._get_sequence_dict()

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            seq_per_class.setdefault(class_name, []).append(seq_id)
        return seq_per_class

    def get_name(self):
        return 'Visevent'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        return {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            return OrderedDict({
                'object_class_name': meta_info[5].split(': ')[-1][:-1],
                'motion_class': meta_info[6].split(': ')[-1][:-1],
                'major_class': meta_info[7].split(': ')[-1][:-1],
                'root_class': meta_info[8].split(': ')[-1][:-1],
                'motion_adverb': meta_info[9].split(': ')[-1][:-1],
            })
        except Exception:
            return OrderedDict({
                'object_class_name': None,
                'motion_class': None,
                'major_class': None,
                'root_class': None,
                'motion_adverb': None,
            })

    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            seq_per_class.setdefault(object_class, []).append(i)
        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        dir_list = [name for name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, name))]
        dir_list.sort()
        return dir_list

    def _get_sequence_dict(self):
        seq_dict = {'visible': {}, 'infrared': {}}
        for seq in self.sequence_list:
            seq_dict['visible'][seq] = sorted(os.listdir(os.path.join(self.root, seq, 'vis_imgs')))
            seq_dict['infrared'][seq] = sorted(os.listdir(os.path.join(self.root, seq, 'event_imgs')))
        return seq_dict

    def _read_bb_anno(self, seq_path):
        gt = pandas.read_csv(
            os.path.join(seq_path, 'groundtruth.txt'),
            delimiter=',',
            header=None,
            dtype=np.float32,
            na_filter=False,
            low_memory=False,
        ).values
        tgt = pandas.read_csv(
            os.path.join(seq_path, 'groundtruth.txt'),
            delimiter=',',
            header=None,
            dtype=np.float32,
            na_filter=False,
            low_memory=False,
        ).values
        return torch.tensor(gt), torch.tensor(tgt)

    def _read_target_visible(self, seq_path):
        occlusion_file = os.path.join(seq_path, 'absence.label')
        cover_file = os.path.join(seq_path, 'cover.label')

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover > 0).byte()
        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        rgbbbox, tbbox = self._read_bb_anno(seq_path)
        valid = (rgbbbox[:, 2] > 5.0) & (rgbbbox[:, 3] > 5.0)
        visible = valid
        return {'bbox': rgbbbox, 'valid': valid, 'visible': visible, 'tbox': tbbox}

    def _get_frame_path(self, seq_path, frame_id):
        seq = os.path.basename(seq_path)
        img_vis_path = os.path.join(seq_path, 'vis_imgs', self.seq_img_dict['visible'][seq][frame_id])
        img_tir_path = os.path.join(seq_path, 'event_imgs', self.seq_img_dict['infrared'][seq][frame_id])
        if not os.path.exists(img_tir_path):
            img_tir_path = img_vis_path
        return img_vis_path, img_tir_path

    def _get_rgb_frame(self, seq_path, frame_id):
        img_vis_path, _ = self._get_frame_path(seq_path, frame_id)
        assert os.path.exists(img_vis_path), 'rgb image {} not exist'.format(img_vis_path)
        return self.image_loader(img_vis_path)

    def _get_tir_frame(self, seq_path, frame_id):
        _, img_tir_path = self._get_frame_path(seq_path, frame_id)
        assert os.path.exists(img_tir_path), 't image {} not exist'.format(img_tir_path)
        return self.image_loader(img_tir_path)

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)
        object_meta = OrderedDict({
            'object_class_name': obj_class,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None,
        })

        rgb_frame_list = [self._get_rgb_frame(seq_path, f_id) for f_id in frame_ids]
        tir_frame_list = [self._get_tir_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {key: [value[f_id, ...].clone() for f_id in frame_ids] for key, value in anno.items()}
        return rgb_frame_list, tir_frame_list, anno_frames, object_meta

    def _get_class(self, seq_path):
        return os.path.basename(seq_path)
