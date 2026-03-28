import glob
import os

import numpy as np

from lib.test.evaluation.data import Sequence_RGBT, BaseDataset, SequenceList


def RGBT234Dataset():
    return RGBT234DatasetClass()


class RGBT234DatasetClass(BaseDataset):
    """RGBT234 evaluation dataset."""

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbt234_path
        if not self.base_path:
            raise RuntimeError('RGBT234 dataset path is not configured. Please set `rgbt234_path` in lib/test/evaluation/local.py.')
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, sequence_name, 'visible.txt')
        try:
            ground_truth_rect = np.loadtxt(anno_path, dtype=np.float64)
        except Exception:
            ground_truth_rect = np.loadtxt(anno_path, delimiter=',', dtype=np.float64)

        anno_pathi = os.path.join(self.base_path, sequence_name, 'infrared.txt')
        try:
            ground_truth_recti = np.loadtxt(anno_pathi, dtype=np.float64)
        except Exception:
            ground_truth_recti = np.loadtxt(anno_pathi, delimiter=',', dtype=np.float64)

        frames = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'visible', '*.jpg')))
        framesi = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'infrared', '*.jpg')))
        return Sequence_RGBT(sequence_name, frames, framesi, 'RGBT234', ground_truth_rect, ground_truth_recti)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = [name for name in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, name))]
        sequence_list.sort()
        return sequence_list
