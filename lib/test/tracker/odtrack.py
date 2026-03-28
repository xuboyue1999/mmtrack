import math
import numpy as np

from lib.models.odtrack import build_odtrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target

import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

class ODTrack(BaseTracker):
    def __init__(self, params):
        super(ODTrack, self).__init__(params)
        network = build_odtrack(params.cfg, training=False)


        ckpt = torch.load(self.params.checkpoint, weights_only=False, map_location="cpu")['net']
        network.load_state_dict(ckpt, strict=True)

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()


        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:

                self._init_visdom(None, 1)

        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, tir, info: dict):

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)


        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(tir, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr_tir = z_patch_arr_tir
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)

        with torch.no_grad():

            self.memory_frames = [template.tensors]

            self.memory_frames_tir = [template_tir.tensors]

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))


        self.memory_masks_tir = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox_tir = self.transform_bbox_to_crop(info['init_bbox'], resize_factor_tir,
                                                        template_tir.tensors.device).squeeze(1)
            self.memory_masks_tir.append(generate_mask_cond(self.cfg, 1, template_tir.tensors.device, template_bbox_tir))


        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, tir, info: dict = None):

        H, W, _ = image.shape

        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)


        search = self.preprocessor.process(x_patch_arr, x_amask_arr)


        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(tir, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)


        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()

            template_list_tir = self.memory_frames_tir.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = torch.cat(self.memory_masks, dim=1)
                box_mask_z_tir = torch.cat(self.memory_masks_tir, dim=1)

        else:
            template_list, box_mask_z = self.select_memory_frames()
            template_list_tir, box_mask_z_tir = self.select_memory_frames_tir()


        with torch.no_grad():


            out_dict = self.network.forward(template=template_list, template_tir=template_list_tir, search=[search.tensors], search_tir=[search_tir.tensors],ce_template_mask=box_mask_z)
        if isinstance(out_dict, list):
            out_dict = out_dict[-1]


        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)


        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()


        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)


        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors


        z_patch_arr_tir, z_resize_factor_tir, z_amask_arr_tir = sample_target(tir, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
        frame_tir = cur_frame_tir.tensors


        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()


            frame_tir = frame_tir.detach().cpu()

        self.memory_frames.append(frame)
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))


        self.memory_frames_tir.append(frame_tir)
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox_tir = self.transform_bbox_to_crop(self.state, z_resize_factor, frame_tir.device).squeeze(1)
            self.memory_masks_tir.append(generate_mask_cond(self.cfg, 1, frame_tir.device, template_bbox_tir))

        if 'pred_iou' in out_dict.keys():
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)


        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:


                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()

            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:

            return {"target_bbox": self.state}

    def select_memory_frames(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames, select_masks = [], []

        for idx in indexes:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)

            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                select_masks.append(box_mask_z.cuda())

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1)
        else:
            return select_frames, None

    def select_memory_frames_tir(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames_tir, select_masks_tir = [], []

        for idx in indexes:
            frames_tir = self.memory_frames_tir[idx]
            if not frames_tir.is_cuda:
                frames_tir = frames_tir.cuda()
            select_frames_tir.append(frames_tir)

            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z_tir = self.memory_masks_tir[idx]
                select_masks_tir.append(box_mask_z_tir.cuda())

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames_tir, torch.cat(select_masks_tir, dim=1)
        else:
            return select_frames_tir, None

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(

                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return ODTrack
