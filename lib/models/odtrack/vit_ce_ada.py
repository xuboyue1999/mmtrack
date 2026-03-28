import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks_ada import CEBlock

_logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerMemoryNetwork(nn.Module):
    def __init__(self, token_dim, stm_size, ltm_size, pm_size):
        super(MultiLayerMemoryNetwork, self).__init__()
        self.token_dim = token_dim


        self.register_buffer('stm', torch.zeros(stm_size, token_dim))
        self.register_buffer('ltm', torch.zeros(ltm_size, token_dim))
        self.register_buffer('pm', torch.zeros(pm_size, token_dim))


    def update_memory(self, inputs):

        device = inputs.device


        self.stm = self.stm.to(device)
        self.ltm = self.ltm.to(device)
        self.pm = self.pm.to(device)

        new_memory = inputs.squeeze(1).to(device)


        self.stm = torch.cat((new_memory, self.stm[:-new_memory.size(0)].clone().to(device)), dim=0)
        query = self.ltm.clone()
        key = self.stm.clone()
        value = self.stm.clone()


        attention_scores = torch.matmul(query, key.transpose(0, 1)) / (self.token_dim ** 0.5)
        attention_weights = F.softmax(attention_scores.clone(), dim=-1)


        new_ltm = torch.matmul(attention_weights, value)


        self.ltm = self.ltm.detach() * 0.5 + new_ltm.detach() * 0.5


        query = self.pm.clone()
        key = self.ltm.clone()
        value = self.ltm.clone()


        attention_scores = torch.matmul(query, key.transpose(0, 1)) / (self.token_dim ** 0.5)
        attention_weights = F.softmax(attention_scores.clone(), dim=-1)


        new_pm = torch.matmul(attention_weights, value)


        self.pm = self.pm.detach() * 0.5 + new_pm.detach() * 0.5
    def forward(self, query):

        query=query.squeeze(1)
        device = query.device


        self.stm = self.stm.to(device)
        self.ltm = self.ltm.to(device)
        self.pm = self.pm.to(device)

        stm_scores = torch.matmul(self.stm.clone(), query.T.clone())
        ltm_scores = torch.matmul(self.ltm.clone(), query.T.clone())
        pm_scores = torch.matmul(self.pm.clone(), query.T.clone())

        stm_weights = F.softmax(stm_scores.clone(), dim=0).T
        ltm_weights = F.softmax(ltm_scores.clone(), dim=0).T
        pm_weights = F.softmax(pm_scores.clone(), dim=0).T

        combined_response = (
                torch.matmul(stm_weights, self.stm.clone()) +
                torch.matmul(ltm_weights, self.ltm.clone()) +
                torch.matmul(pm_weights, self.pm.clone())
        )

        combined_response = combined_response.to(device)
        output = combined_response
        output = output.unsqueeze(1)

        return output


class Feature_fusion(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.FF_layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU())),
            ('fc2_RGB', nn.Sequential(
                nn.Linear(hidden_dim, feat_dim),
                nn.Sigmoid())),
            ('fc2_T', nn.Sequential(
                nn.Linear(hidden_dim, feat_dim),
                nn.Sigmoid())),

        ]))

    def forward(self, feat_RGB, feat_T):
        feat_sum_1 = self.GAP(feat_RGB + feat_T)

        feat_sum_1 = feat_sum_1.view(-1, feat_sum_1.shape[1])

        feat_sum_1 = self.FF_layers.fc1(feat_sum_1)
        w_RGB = self.FF_layers.fc2_RGB(feat_sum_1)
        w_T = self.FF_layers.fc2_T(feat_sum_1)

        r11 = feat_RGB * w_RGB.view(-1, self.feat_dim, 1, 1)
        t11 = feat_T * w_T.view(-1, self.feat_dim, 1, 1)
        return r11, t11
class spec_select(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(spec_select, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        self.act2 = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))

        self.modulate = Feature_fusion(inchannels)

        self.modulate2 = Feature_fusion(inchannels)
        self.forve=Fovea_Fusion2d(inplanes=inchannels, hide_channel=16, smooth=True,need_reshape=False)
    def forward(self, R,T):
        input_R = R
        input_T = T
        low_filter_R = self.ap(R)
        low_filter_R = self.conv(low_filter_R)
        low_filter_R = self.bn(low_filter_R)
        low_filter_T = self.ap(T)
        low_filter_T = self.conv(low_filter_T)
        low_filter_T = self.bn(low_filter_T)
        n, c, h, w = R.shape
        R = F.unfold(self.pad(R), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter_R.shape
        low_filter_R = low_filter_R.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter_R = self.act(low_filter_R)
        low_part_R = torch.sum(R * low_filter_R, dim=3).reshape(n, c, h, w)


        T = F.unfold(self.pad(T), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        low_filter_T = low_filter_T.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter_T = self.act2(low_filter_T)
        low_part_T = torch.sum(T * low_filter_T, dim=3).reshape(n, c, h, w)

        high_part_T =input_T - low_part_T

        high_part_R = input_R - low_part_R
        R_l,R_h = self.modulate(low_part_R, high_part_R)
        T_l,T_h = self.modulate2(low_part_T, high_part_T)
        T=self.forve(R_l+R_h,T_l+T_h)
        return input_R,T

class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''


        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x

        return output
class Fovea_Fusion2d(nn.Module):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False,need_reshape=True):
        super(Fovea_Fusion2d, self).__init__()
        self.inplanes=inplanes
        self.hide_channel=hide_channel
        self.convR = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.convT = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.convG = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.convO = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.foveaR = Fovea(smooth=smooth)
        self.foveaT = Fovea(smooth=smooth)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.L = nn.Linear(hide_channel, hide_channel)
        self.sigmoid = nn.Sigmoid()
        self.need_reshape=need_reshape
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, R,T):
        if self.need_reshape:
            R= R.permute(0, 2, 1)
            T = T.permute(0, 2, 1)
        if R.shape[2]!=T.shape[2]:
            T=F.interpolate(T,(R.shape[2],R.shape[3]))
        R=self.convR(R)
        T=self.convT(T)

        G=R+T
        G_f=self.GAP(G)
        G_f = G_f.view(-1, G_f.shape[1])
        G_f=self.L(G_f)
        w_G=self.sigmoid(G_f)
        G=G*w_G.view(-1, self.hide_channel, 1,1)


        R=self.foveaR(R)
        T=self.foveaT(T)
        fused=R+T+G
        fused=self.convO(fused)

        if self.need_reshape:
            fused=fused.permute(0, 2,  1)
        return fused

class Fovea_Fusion(nn.Module):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False, need_reshape=True):
        super(Fovea_Fusion, self).__init__()
        self.inplanes = inplanes
        self.hide_channel = hide_channel
        self.convR = nn.Conv1d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.convT = nn.Conv1d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.convG = nn.Conv1d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.convO = nn.Conv1d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.foveaR = Fovea(smooth=smooth)
        self.foveaT = Fovea(smooth=smooth)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.L = nn.Linear(hide_channel, hide_channel)
        self.sigmoid = nn.Sigmoid()
        self.need_reshape = need_reshape
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, R, T):
        if self.need_reshape:
            R = R.permute(0, 2, 1)
            T = T.permute(0, 2, 1)
        if R.shape[2] != T.shape[2]:

            T = F.interpolate(T, R.shape[2])
        R = self.convR(R)
        T = self.convT(T)

        G = R + T
        G_f = self.GAP(G)
        G_f = G_f.view(-1, G_f.shape[1])
        G_f = self.L(G_f)
        w_G = self.sigmoid(G_f)
        G = G * w_G.view(-1, self.hide_channel, 1)

        R = self.foveaR(R)
        T = self.foveaT(T)
        fused = R + T + G
        fused = self.convO(fused)
        if self.need_reshape:
            fused = fused.permute(0, 2, 1)
        return fused


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, add_cls_token=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_T = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.add_cls_token = add_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        bottle_adapter = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )
            bottle_adapter.append(Fovea_Fusion(inplanes=embed_dim, hide_channel=16, smooth=True))

        self.blocks = nn.Sequential(*blocks)
        self.bottle_adapters = nn.Sequential(*bottle_adapter)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

        self.input_adapter = spec_select(inchannels=embed_dim)

        token_dim = 768
        stm_size = 8
        ltm_size = 8
        pm_size = 3
        self.memory_adapter = MultiLayerMemoryNetwork(token_dim, stm_size, ltm_size, pm_size)
    def forward_features(self, z, x, z_t=None, x_t=None, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, track_query=None,
                         token_type="add", token_len=1
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape

        z = z.flatten(0, 1)

        z = self.patch_embed(z)

        x_t = self.patch_embed_T(x_t)
        z_t = torch.stack(z_t, dim=1)
        z_t = z_t.flatten(0, 1)
        z_t = self.patch_embed_T(z_t)
        x = token2feature(x)
        x_t = token2feature(x_t)
        z = token2feature(z)
        z_t = token2feature(z_t)
        x, x_t = self.input_adapter(x, x_t)
        z, z_t = self.input_adapter(z, z_t)
        x = x + x_t
        z = z + z_t
        x = feature2token(x)
        z = feature2token(z)
        x_t = feature2token(x_t)
        z_t = feature2token(z_t)


        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            if token_type == "concat":
                if track_query is None:
                    query = self.cls_token.expand(B, token_len, -1)
                else:
                    track_len = track_query.size(1)
                    new_query = self.cls_token.expand(B, token_len - track_len, -1)
                    query = torch.cat([new_query, track_query], dim=1)
            elif token_type == "add":
                new_query = self.cls_token.expand(B, token_len, -1)
                query = new_query if track_query is None else track_query + new_query


            self.memory_adapter.update_memory(query)
            query = self.memory_adapter(query)
            query = query + self.cls_pos_embed


        z = z + self.pos_embed_z
        x = x + self.pos_embed_x


        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed

        if T_z > 1:
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            z = z.flatten(1, 2)
            z_t = z_t.view(B, T_z, -1, z_t.size()[-1]).contiguous()
            z_t = z_t.flatten(1, 2)

        lens_z = z.shape[1]
        lens_x = x.shape[1]

        x = combine_tokens(z, x, mode=self.cat_mode)
        x_t = combine_tokens(z_t, x_t, mode=self.cat_mode)

        if self.add_cls_token:
            x = torch.cat([query, x], dim=1)
            query_len = query.size(1)

            x_t = torch.cat([query, x_t], dim=1)
            query_len = query.size(1)

        x = self.pos_drop(x)
        x_t = self.pos_drop(x_t)
        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []

        for i, blks in enumerate(zip(self.blocks, self.bottle_adapters)):
            blk = blks[0]
            adp = blks[1]
            if self.add_cls_token:

                x, global_index_t, global_index_s, removed_index_s, attn =\
                    blk(x, x_t, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token, query_len=query_len)


                x_t = adp(x, x_t)
                x = x + x_t
            else:
                x, global_index_t, global_index_s, removed_index_s, attn =\
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        if self.add_cls_token:
            query = x[:, :query_len]
            z = x[:, query_len:lens_z_new + query_len]
            x = x[:, lens_z_new + query_len:]
        else:
            z = x[:, :lens_z_new]
            x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)

            C = x.shape[-1]

            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                             src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)


        x = torch.cat([query, z, x], dim=1)


        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
        }
        return x, aux_dict

    def forward(self, z, x, z_t=None, x_t=None, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, track_query=None,
                token_type="add", token_len=1):
        x, aux_dict = self.forward_features(z, x, z_t, x_t, ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            track_query=track_query, token_type=token_type, token_len=token_len)
        return x, aux_dict


def _create_vision_transformer_ada(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            try:
                checkpoint = torch.load(pretrained, map_location="cpu")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
                print('Load pretrained model from: ' + pretrained)
            except:
                print("Warning: MAE Pretrained model weights are not loaded !")

    return model


def vit_base_patch16_224_ce_ada(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer_ada(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_ada(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer_ada(pretrained=pretrained, **model_kwargs)
    return model