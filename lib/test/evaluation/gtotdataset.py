import glob
import os

import numpy as np

from lib.test.evaluation.data import Sequence_RGBT, BaseDataset, SequenceList


def GTOTDataset():
    return GTOTDatasetClass()


class GTOTDatasetClass(BaseDataset):
    """GTOT evaluation dataset."""

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.gtot_path
        if not self.base_path:
            raise RuntimeError('GTOT dataset path is not configured. Please set `gtot_path` in lib/test/evaluation/local.py.')
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, sequence_name, 'groundTruth_v.txt')
        try:
            ground_truth_rect = np.loadtxt(anno_path, dtype=np.float64)
        except Exception:
            ground_truth_rect = np.loadtxt(anno_path, delimiter=',', dtype=np.float64)
        ground_truth_rect[:, 2] = ground_truth_rect[:, 2] - ground_truth_rect[:, 0]
        ground_truth_rect[:, 3] = ground_truth_rect[:, 3] - ground_truth_rect[:, 1]

        anno_pathi = os.path.join(self.base_path, sequence_name, 'groundTruth_i.txt')
        try:
            ground_truth_recti = np.loadtxt(anno_pathi, dtype=np.float64)
        except Exception:
            ground_truth_recti = np.loadtxt(anno_pathi, delimiter=',', dtype=np.float64)
        ground_truth_recti[:, 2] = ground_truth_recti[:, 2] - ground_truth_recti[:, 0]
        ground_truth_recti[:, 3] = ground_truth_recti[:, 3] - ground_truth_recti[:, 1]

        frames = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'v', '*.png')))
        if not frames:
            frames = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'v', '*.bmp')))

        framesi = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'i', '*.png')))
        if not framesi:
            framesi = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'i', '*.bmp')))

        return Sequence_RGBT(sequence_name, frames, framesi, 'GTOT', ground_truth_rect, ground_truth_recti)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = [name for name in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, name))]
        sequence_list.sort()
        return sequence_list
