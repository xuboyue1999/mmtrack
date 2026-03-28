from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

import sys
import time
import os
from lib.test.evaluation import Tracker
import lib.test.vot.vot as vot
from lib.test.vot.vot22_utils import *
from lib.train.dataset.depth_utils import get_rgbd_frame


class STITrackv2(object):
    def __init__(self, tracker_name='', para_name='',datasetname="depthtrack"):

        tracker_info = Tracker(tracker_name, para_name, datasetname, None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def write(self, str):
        txt_path = ""
        file = open(txt_path, 'a')
        file.write(str)

    def initialize(self, img_rgb,x, selection):


        x, y, w, h = selection
        bbox = [x,y,w,h]
        self.H, self.W, _ = img_rgb.shape
        init_info = {'init_bbox': bbox}
        _ = self.tracker.initialize(img_rgb,x, init_info)

    def track(self, img_rgb,x):

        outputs = self.tracker.track(img_rgb,x)
        pred_bbox = outputs['target_bbox']

        max_score = 1
        return pred_bbox, max_score


def run_vot_exp(tracker_name, para_name, datasetname="depthtrack",vis=False, out_conf=False, channel_type='color'):

    torch.set_num_threads(1)
    save_root = os.path.join('', para_name)
    if vis and (not os.path.exists(save_root)):
        os.mkdir(save_root)
    tracker = STITrackv2(tracker_name=tracker_name, para_name=para_name, datasetname=datasetname)

    if channel_type=='rgb':
        channel_type=None
    handle = vot.VOT("rectangle", channels=channel_type)

    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root,seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


    if isinstance(imagefile, list) and len(imagefile)==2:

        image = cv2.cvtColor(cv2.imread(imagefile[0]), cv2.COLOR_BGR2RGB)
        dp = cv2.imread(imagefile[1], -1)
        max_depth = min(np.median(dp) * 3, 10000)
        dp[dp > max_depth] = max_depth
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        x = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
    else:
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

    tracker.initialize(image,x, selection)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break


        if isinstance(imagefile, list) and len(imagefile) == 2:
            image = cv2.cvtColor(cv2.imread(imagefile[0]), cv2.COLOR_BGR2RGB)
            dp = cv2.imread(imagefile[1], -1)
            max_depth = min(np.median(dp) * 3, 10000)
            dp[dp > max_depth] = max_depth
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            x = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
        else:
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

        b1, max_score = tracker.track(image,x)

        if out_conf:
            handle.report(vot.Rectangle(*b1), max_score)
        else:
            handle.report(vot.Rectangle(*b1))
        if vis:
            '''Visualization'''

            image_ori = image[:,:,::-1].copy()
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg','_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
