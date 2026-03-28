import os
import cv2
import numpy as np
import argparse
import importlib
import multiprocessing
import torch
import time

from lib.test.evaluation.environment import env_settings


def genConfig(seq_path, set_type):
    if set_type != 'VisEvent':
        raise ValueError('Unsupported dataset: {}'.format(set_type))

    rgb_img_list = sorted(
        os.path.join(seq_path, 'vis_imgs', p)
        for p in os.listdir(os.path.join(seq_path, 'vis_imgs'))
        if os.path.splitext(p)[1] == '.bmp'
    )
    e_img_list = sorted(
        os.path.join(seq_path, 'event_imgs', p)
        for p in os.listdir(os.path.join(seq_path, 'event_imgs'))
        if os.path.splitext(p)[1] == '.bmp'
    )
    rgb_gt = np.loadtxt(os.path.join(seq_path, 'groundtruth.txt'), delimiter=',')
    absent_label = np.loadtxt(os.path.join(seq_path, 'absent_label.txt'))
    return rgb_img_list, e_img_list, rgb_gt, absent_label


def get_parameters(script_name, yaml_name):
    param_module = importlib.import_module('lib.test.parameter.{}'.format(script_name))
    return param_module.parameters(yaml_name)


def create_tracker(script_name, params, dataset_name):
    tracker_module = importlib.import_module('lib.test.tracker.{}'.format(script_name))
    tracker_class = tracker_module.get_tracker_class()
    return tracker_class(params)


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, debug=0, script_name='prompt'):
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except Exception:
        pass

    save_name = '{}_{}final_test2'.format(script_name, yaml_name)
    save_folder = os.path.join('.', 'RGBE_workspace', 'results', dataset_name, save_name)
    save_path = os.path.join(save_folder, '{}.txt'.format(seq_name))
    os.makedirs(save_folder, exist_ok=True)

    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    params = get_parameters(script_name, yaml_name)
    if debug is None:
        debug = getattr(params, 'debug', 0)
    params.debug = debug

    mmtrack = create_tracker(script_name, params, 'visevent')
    tracker = TRACK_RGBE(tracker=mmtrack)

    seq_path = os.path.join(seq_home, seq_name)
    print('Processing sequence: {}'.format(seq_name))
    rgb_img_list, e_img_list, rgb_gt, absent_label = genConfig(seq_path, dataset_name)

    if absent_label[0] == 0:
        first_present_idx = absent_label.argmax()
        rgb_img_list = rgb_img_list[first_present_idx:]
        e_img_list = e_img_list[first_present_idx:]
        rgb_gt = rgb_gt[first_present_idx:]

    if len(rgb_img_list) == len(rgb_gt):
        result = np.zeros_like(rgb_gt)
    else:
        result = np.zeros((len(rgb_img_list), 4), dtype=rgb_gt.dtype)

    result[0] = np.copy(rgb_gt[0])
    toc = 0

    for frame_idx, (rgb_path, e_path) in enumerate(zip(rgb_img_list, e_img_list)):
        tic = cv2.getTickCount()
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        e = cv2.imread(e_path)
        e = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            tracker.initialize(rgb, e, rgb_gt[0].tolist())
        else:
            region, _ = tracker.track(rgb, e)
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class TRACK_RGBE(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, e, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, e, init_info)

    def track(self, img_RGB, e):
        outputs = self.tracker.track(img_RGB, e)
        pred_bbox = outputs['target_bbox']
        pred_score = 0
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBE dataset.')
    parser.add_argument('--script_name', type=str, default='odtrack', help='Name of tracking method(seqtrackv2).')
    parser.add_argument('--yaml_name', type=str, default='baseline_rgbe', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='VisEvent', help='Name of dataset (VisEvent).')
    parser.add_argument('--threads', default=0, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--mode', default='sequential', type=str, help='running mode: [sequential , parallel]')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    seq_home = env_settings().visevent_path
    if dataset_name != 'VisEvent':
        raise ValueError('Error dataset!')
    if not seq_home:
        raise RuntimeError('VisEvent dataset path is not configured. Please set `visevent_path` in lib/test/evaluation/local.py.')

    seq_list = os.listdir(seq_home)
    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug, args.script_name) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug, args.script_name) for s in seq_list]
        for sequence in sequence_list:
            run_sequence(*sequence)
    print(f"Totally cost {time.time() - start} seconds!")
