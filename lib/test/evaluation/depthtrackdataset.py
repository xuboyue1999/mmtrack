import glob
import os

import numpy as np

from lib.test.evaluation.data import Sequence_RGBT, BaseDataset, SequenceList


def depthtrackDataset():
    return DepthtrackDatasetClass()


class DepthtrackDatasetClass(BaseDataset):
    """DepthTrack evaluation dataset."""

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.depthtrack_path
        if not self.base_path:
            raise RuntimeError('DepthTrack dataset path is not configured. Please set `depthtrack_path` in lib/test/evaluation/local.py.')
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        try:
            ground_truth_rect = np.loadtxt(anno_path, dtype=np.float64)
        except Exception:
            ground_truth_rect = np.loadtxt(anno_path, delimiter=',', dtype=np.float64)

        anno_pathi = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        try:
            ground_truth_recti = np.loadtxt(anno_pathi, dtype=np.float64)
        except Exception:
            ground_truth_recti = np.loadtxt(anno_pathi, delimiter=',', dtype=np.float64)

        frames = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'color', '*.jpg')))
        framesi = sorted(glob.glob(os.path.join(self.base_path, sequence_name, 'depth', '*.png')))
        return Sequence_RGBT(sequence_name, frames, framesi, 'depthtrack', ground_truth_rect, ground_truth_recti)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        for candidate in (
            os.path.join(self.base_path, 'list.txt'),
            os.path.abspath(os.path.join(self.base_path, '..', 'list.txt')),
        ):
            if os.path.exists(candidate):
                with open(candidate, 'r') as list_file:
                    return [line.strip() for line in list_file if line.strip()]

        sequence_list = [name for name in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, name))]
        sequence_list.sort()
        return sequence_list
