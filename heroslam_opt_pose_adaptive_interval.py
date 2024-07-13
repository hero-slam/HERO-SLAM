import os
# os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2
import time
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe_adaptive_interval import KeyFrameDatabase
from model.utils import compute_loss
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation, align, evaluate
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, \
    matrix_to_quaternion
from sp_lg.lightglue import LightGlue
from sp_lg.superpoint import SuperPoint
from sp_lg.disk import DISK
from sp_lg.utils import load_image, match_pair
from sp_lg import viz2d
from pathlib import Path
from model.loss import SSIM


class HeroSLAM():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)

        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)

        # SuperPoint+LightGlue
        self.sp_extractor = SuperPoint(max_num_keypoints=self.config['mapping']['sample']).eval().to(self.device)
        match_conf = {
            'width_confidence': 0.99,  # for point pruning
            'depth_confidence': 0.95,  # for early stopping,
        }
        self.lg_matcher = LightGlue(pretrained='superpoint', **match_conf).eval().to(self.device)

        self.last_image = None
        self.last_image_kp = None
        self.last_image_pose = None
        self.last_image_gt_pose = None
        self.last_image_depth = None
        self.last_image_frame_id = None

        self.track_img_num = 0

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')

        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError

    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.gt_c2w_data = {}
        # self.load_gt_pose()

    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(
            self.device)

    def create_kf_database(self, config):
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config,
                                self.dataset.H,
                                self.dataset.W,
                                num_kf,
                                self.dataset.num_rays_to_save,
                                self.device)

    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'pose_gt': self.gt_c2w_data,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']
        self.gt_c2w_data = dict['pose_gt']

    def select_samples(self, H, W, samples, rgb):
        '''
        randomly select samples from the image
        '''
        
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)

        if self.last_image is None:
            img = torch.tensor(rgb.clone().permute(0, 3, 1, 2), dtype=torch.float).to(self.device)
            feats, descriptors_all = self.sp_extractor({'image': img})
            self.last_image_kp = feats
            self.last_image = rgb.clone()
            self.last_image_feature = descriptors_all

        return indice

    def select_samples_from_sp(self, H, W, samples, rgb, batch=None):
        '''
        select samples from the image using superpoint heatmap
        '''

        assert H == rgb.shape[1]
        assert W == rgb.shape[2]

        img = torch.tensor(rgb.permute(0, 3, 1, 2), dtype=torch.float).to(self.device)
        feats, desc_all = self.sp_extractor({'image': img})

        last_img = torch.tensor(self.last_image.permute(0, 3, 1, 2), dtype=torch.float).to(self.device)

        # match prev_prev_image and current_image
        data = {'image0': img, 'image1': last_img}
        pred = {**{k + '0': v for k, v in feats.items()},
                **{k + '1': v for k, v in self.last_image_kp.items()},
                **data}
        pred = {**pred, **self.lg_matcher(pred)}
        pred = {k: v.to(self.device).detach()[0] if
        isinstance(v, torch.Tensor) else v for k, v in pred.items()}

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        # create match indices
        matches0, mscores0 = pred['matches0'], pred['matching_scores0']

        valid = matches0 > -1
        matches = torch.stack([torch.where(valid)[0], matches0[valid]], -1)
        pred = {**pred, 'matches': matches, 'matching_scores': mscores0[valid]}

        mscores, indices = mscores0[valid].sort(dim=0, descending=True)
        mscores = mscores[:500]
        indices = indices[:500]
        matches = matches[indices]

        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        m_kpts0 = m_kpts0.detach().cpu().numpy()
        m_kpts1 = m_kpts1.detach().cpu().numpy()
        print("m_kpts0 = ", m_kpts0.shape)
        print("m_kpts1 = ", m_kpts1.shape)

        kpts0 = torch.tensor(np.array(m_kpts0)).to(self.device)
        kpts1 = torch.tensor(np.array(m_kpts1)).to(self.device)
        return kpts0, kpts1, feats, mscores, desc_all

    def get_gt_matches(self, m_kpts0_cand, m_kpts1_cand, batch):
        cxcy = torch.tensor([self.dataset.cx, self.dataset.cy]).to(self.device)[None, ...]
        fxfy = torch.tensor([self.dataset.fx, self.dataset.fy]).to(self.device)[None, ...]

        mkpts_cur = torch.tensor(np.array(m_kpts0_cand)).to(self.device)
        mkpts_last = torch.tensor(np.array(m_kpts1_cand)).to(self.device)
        normalized_kps = (mkpts_cur - cxcy) / fxfy
        normalized_kps = torch.cat([normalized_kps, torch.ones(normalized_kps.shape[0], 1).to(self.device)], dim=-1)

        mkpts_cur_depth = batch['depth'][0, mkpts_cur[:, 1].long(), mkpts_cur[:, 0].long()][..., None].to(self.device)

        mask = (mkpts_cur_depth > 0.1) & (mkpts_cur_depth < self.config['cam']['depth_trunc'])

        normalized_kps = normalized_kps[mask[:, 0]]
        mkpts_cur_depth = mkpts_cur_depth[mask[:, 0]]
        mkpts_last = mkpts_last[mask[:, 0]]
        mkpts_cur = mkpts_cur[mask[:, 0]]
        mscores = mscores[mask[:, 0]]

        pts_cur = (normalized_kps * mkpts_cur_depth).to(self.device)

        T_nerf_cam = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape([1, 4, 4]).to(
            self.device)

        T_last_cur = (self.last_image_gt_pose.to(self.device) @ T_nerf_cam).inverse().to(self.device) @ (
                batch['c2w'].to(self.device) @ T_nerf_cam).to(self.device)

        Rp = torch.sum(pts_cur[:, None, :] * T_last_cur[:, :3, :3], -1)
        t = T_last_cur[:, :3, -1].repeat(pts_cur.shape[0], 1)
        pts_last_warp = Rp + t
        pts_last_warp_normalize = pts_last_warp[:, :2] / pts_last_warp[:, -1:]
        pts_last_warp_normalize = pts_last_warp_normalize * fxfy + cxcy

        pt_diff = torch.sqrt(torch.sum(torch.square(pts_last_warp_normalize - mkpts_last), dim=-1))
        mask = (pt_diff < 1.0).detach().clone().cpu().numpy().flatten().astype(bool)
        m_kpts0 = mkpts_cur[mask[:], :]
        m_kpts1 = mkpts_last[mask[:], :]
        mscores = mscores[mask[:]]
        m_kpts0 = m_kpts0.detach().cpu().numpy()
        m_kpts1 = m_kpts1.detach().cpu().numpy()

        return mscores, m_kpts0, m_kpts1

    def get_loss_from_match(self, mkpts_cur, mkpts_last, c2w_est, batch, mscores, iter):

        cxcy = torch.tensor([self.dataset.cx, self.dataset.cy]).to(self.device)[None, ...]
        fxfy = torch.tensor([self.dataset.fx, self.dataset.fy]).to(self.device)[None, ...]

        normalized_kps = (mkpts_cur - cxcy) / fxfy
        normalized_kps = torch.cat([normalized_kps, torch.ones(normalized_kps.shape[0], 1).to(self.device)], dim=-1)

        mkpts_cur_depth = batch['depth'][0, mkpts_cur[:, 1].long(), mkpts_cur[:, 0].long()][..., None].to(self.device)

        mask = (mkpts_cur_depth > 0.1) & (mkpts_cur_depth < self.config['cam']['depth_trunc'])
        normalized_kps = normalized_kps[mask[:, 0]]
        mkpts_cur_depth = mkpts_cur_depth[mask[:, 0]]
        mkpts_last = mkpts_last[mask[:, 0]]
        mscores = mscores[mask[:, 0]]

        pts_cur = (normalized_kps * mkpts_cur_depth).to(self.device)

        T_nerf_cam = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape([1, 4, 4]).to(
            self.device)
        T_last_cur = (self.last_image_pose.to(self.device) @ T_nerf_cam).inverse().to(self.device) @ (
                    c2w_est @ T_nerf_cam)

        Rp = torch.sum(pts_cur[:, None, :] * T_last_cur[:, :3, :3], -1)
        t = T_last_cur[:, :3, -1].repeat(pts_cur.shape[0], 1)
        pts_last_warp = Rp + t

        pts_last_warp_normalize = pts_last_warp[:, :2] / pts_last_warp[:, -1:]
        pts_last_warp_normalize = pts_last_warp_normalize * fxfy + cxcy

        patch_size = 11
        boundary = (patch_size - 1) / 2
        bound = torch.tensor(
            [[boundary + 20, self.dataset.W - boundary - 20], [boundary + 20, self.dataset.H - boundary - 20]]).to(
            self.device)
        mask = ((pts_last_warp_normalize[:, 0] > bound[0, 0]) &
                (pts_last_warp_normalize[:, 0] < bound[0, 1]) &
                (pts_last_warp_normalize[:, 1] > bound[1, 0]) &
                (pts_last_warp_normalize[:, 1] < bound[1, 1]))
        pts_last_warp_normalize = pts_last_warp_normalize[mask]
        mkpts_last = mkpts_last[mask]
        mscores = mscores[mask]
        mscores = torch.nn.functional.normalize(mscores.float(), p=2, dim=0)

        patch_size = 11
        channel_dim = 3
        pts_last_warp_patch_5_rgb = self.get_patch_rgb(pts_last_warp_normalize, patch_size=patch_size,
                                                       channel_dim=channel_dim).to(self.device)
        pts_last_patch_5_rgb = self.get_patch_rgb(mkpts_last, patch_size=patch_size, channel_dim=channel_dim).to(
            self.device)
        pts_last_warp_patch_5_rgb = pts_last_warp_patch_5_rgb * mscores[..., None, None]
        pts_last_patch_5_rgb = pts_last_patch_5_rgb * mscores[..., None, None]

        patch_size = 11
        channel_dim = 1
        pts_last_warp_patch_5_depth = self.get_patch_depth(pts_last_warp_normalize, patch_size=patch_size,
                                                           channel_dim=channel_dim).to(self.device)
        pts_last_patch_5_depth = self.get_patch_depth(mkpts_last, patch_size=patch_size, channel_dim=channel_dim).to(
            self.device)
        pts_last_warp_patch_5_depth = pts_last_warp_patch_5_depth * mscores[..., None, None]
        pts_last_patch_5_depth = pts_last_patch_5_depth * mscores[..., None, None]

        patch_size = 7
        channel_dim = 3
        pts_last_warp_patch_3_rgb = self.get_patch_rgb(pts_last_warp_normalize, patch_size=patch_size,
                                                       channel_dim=channel_dim).to(self.device)
        pts_last_patch_3_rgb = self.get_patch_rgb(mkpts_last, patch_size=patch_size, channel_dim=channel_dim).to(
            self.device)
        pts_last_warp_patch_3_rgb = pts_last_warp_patch_3_rgb * mscores[..., None, None]
        pts_last_patch_3_rgb = pts_last_patch_3_rgb * mscores[..., None, None]

        patch_size = 1
        channel_dim = self.last_image_feature.shape[-1]
        pts_last_warp_patch_1_feature = self.get_patch_feature(pts_last_warp_normalize, patch_size=patch_size,
                                                       channel_dim=channel_dim).to(self.device)
        pts_last_patch_1_feature = self.get_patch_feature(mkpts_last, patch_size=patch_size, channel_dim=channel_dim).to(
            self.device)
        pts_last_warp_patch_1_feature = pts_last_warp_patch_1_feature * mscores[..., None, None]
        pts_last_patch_1_feature = pts_last_patch_1_feature * mscores[..., None, None]

        ssim_loss_5_rgb = SSIM(5, 3).to(self.device)
        loss_5_rgb = ssim_loss_5_rgb.forward(pts_last_warp_patch_5_rgb[:, None, ...], pts_last_patch_5_rgb).sum()

        ssim_loss_5_depth = SSIM(5, 1).to(self.device)
        loss_5_depth = ssim_loss_5_depth.forward(pts_last_warp_patch_5_depth[:, None, ...],
                                                 pts_last_patch_5_depth).sum()

        ssim_loss_3 = SSIM(3, 3).to(self.device)
        loss_3 = ssim_loss_3.forward(pts_last_warp_patch_3_rgb[:, None, ...], pts_last_patch_3_rgb).sum()

        loss_1_feature = torch.nn.functional.smooth_l1_loss(
                    pts_last_warp_patch_1_feature,
                    pts_last_patch_1_feature,
                    beta=0.1,
                    reduction="mean",
                ) * 1.0

        # # normalize kp
        # pts_last_warp_normalize[:, 0] = pts_last_warp_normalize[:, 0] / self.dataset.W
        # pts_last_warp_normalize[:, 1] = pts_last_warp_normalize[:, 1] / self.dataset.H
        # mkpts_last[:, 0] = mkpts_last[:, 0] / self.dataset.W
        # mkpts_last[:, 1] = mkpts_last[:, 1] / self.dataset.H

        loss_kp_mean = torch.nn.functional.smooth_l1_loss(
            pts_last_warp_normalize * mscores[..., None],
            mkpts_last * mscores[..., None],
            beta=0.1,
            reduction="mean",
        ) * 1.0

        loss_pose = (loss_3 + loss_5_rgb + loss_5_depth * 0.01) * 0.01
        loss_pose += loss_kp_mean
        loss_pose += loss_1_feature

        return loss_pose, loss_kp_mean

    def get_patch_rgb(self, pts_last_warp_normalize, patch_size=11, channel_dim=3):
        batch_patch_uv = (
            pts_last_warp_normalize.clone()
                .view(pts_last_warp_normalize.shape[0], 1, pts_last_warp_normalize.shape[1])
                .repeat(1, patch_size * patch_size, 1)
        )
        offset_kernel = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, patch_size) - patch_size // 2,
                    torch.arange(0, patch_size) - patch_size // 2,
                ),
                dim=-1,
            )
                .view(1, patch_size * patch_size, 2)
                .repeat(batch_patch_uv.shape[0], 1, 1)
                .to(self.device)
        )
        batch_patch_uv = batch_patch_uv + offset_kernel
        batch_patch_uv = batch_patch_uv.view(-1, 2)

        batch_patch_uv = batch_patch_uv.reshape([1, 1, -1, 2])
        batch_patch_uv[..., 0] = batch_patch_uv[..., 0] / self.dataset.W * 2.0 - 1.0
        batch_patch_uv[..., 1] = batch_patch_uv[..., 1] / self.dataset.H * 2.0 - 1.0

        pts_last_warp_rgb = torch.nn.functional.grid_sample(
            self.last_image.permute(0, 3, 1, 2).float().to(self.device),
            batch_patch_uv,
            padding_mode="border",
            mode='bicubic'
        ).permute(0, 2, 3, 1).reshape([-1, channel_dim])

        pts_last_warp_patch_rgb = pts_last_warp_rgb.reshape([-1, patch_size * patch_size, channel_dim]).to(self.device)

        return pts_last_warp_patch_rgb

    def get_patch_feature(self, pts_last_warp_normalize, patch_size=11, channel_dim=3):
        batch_patch_uv = (
            pts_last_warp_normalize.clone()
                .view(pts_last_warp_normalize.shape[0], 1, pts_last_warp_normalize.shape[1])
                .repeat(1, patch_size * patch_size, 1)
        )
        offset_kernel = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, patch_size) - patch_size // 2,
                    torch.arange(0, patch_size) - patch_size // 2,
                ),
                dim=-1,
            )
                .view(1, patch_size * patch_size, 2)
                .repeat(batch_patch_uv.shape[0], 1, 1)
                .to(self.device)
        )
        batch_patch_uv = batch_patch_uv + offset_kernel
        batch_patch_uv = batch_patch_uv.view(-1, 2)

        batch_patch_uv = batch_patch_uv.reshape([1, 1, -1, 2])
        batch_patch_uv[..., 0] = batch_patch_uv[..., 0] / self.dataset.W * 2.0 - 1.0
        batch_patch_uv[..., 1] = batch_patch_uv[..., 1] / self.dataset.H * 2.0 - 1.0

        pts_last_warp_feature = torch.nn.functional.grid_sample(
            self.last_image_feature.permute(0, 3, 1, 2).float().to(self.device),
            batch_patch_uv,
            padding_mode="border",
            mode='bicubic'
        ).permute(0, 2, 3, 1).reshape([-1, channel_dim])

        pts_last_warp_patch_feature = pts_last_warp_feature.reshape([-1, patch_size * patch_size, channel_dim]).to(self.device)

        return pts_last_warp_patch_feature

    def get_patch_depth(self, pts_last_warp_normalize, patch_size=11, channel_dim=1):
        batch_patch_uv = (
            pts_last_warp_normalize.clone()
                .view(pts_last_warp_normalize.shape[0], 1, pts_last_warp_normalize.shape[1])
                .repeat(1, patch_size * patch_size, 1)
        )
        offset_kernel = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, patch_size) - patch_size // 2,
                    torch.arange(0, patch_size) - patch_size // 2,
                ),
                dim=-1,
            )
                .view(1, patch_size * patch_size, 2)
                .repeat(batch_patch_uv.shape[0], 1, 1)
                .to(self.device)
        )
        batch_patch_uv = batch_patch_uv + offset_kernel
        batch_patch_uv = batch_patch_uv.view(-1, 2)

        batch_patch_uv = batch_patch_uv.reshape([1, 1, -1, 2])
        batch_patch_uv[..., 0] = batch_patch_uv[..., 0] / self.dataset.W * 2.0 - 1.0
        batch_patch_uv[..., 1] = batch_patch_uv[..., 1] / self.dataset.H * 2.0 - 1.0

        pts_last_warp_depth = torch.nn.functional.grid_sample(
            self.last_image_depth[..., None].permute(0, 3, 1, 2).float().to(self.device),
            batch_patch_uv,
            padding_mode="border",
            mode='bicubic'
        ).permute(0, 2, 3, 1).reshape([-1, channel_dim])

        pts_last_warp_patch_depth = pts_last_warp_depth.reshape([-1, patch_size * patch_size, channel_dim]).to(
            self.device)

        return pts_last_warp_patch_depth

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss += self.config['training']['fs_weight'] * ret["fs_loss"]

        if smooth and self.config['training']['smooth_weight'] > 0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'],
                                                                               self.config['training']['smooth_vox'],
                                                                               margin=self.config['training'][
                                                                                   'smooth_margin'])

        return loss

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        '''
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.gt_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            print('first_frame_mapping {}'.format(i))

            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'], batch['rgb'])

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3],
                               -1)  # transform ray direction from camera coor to world coor

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()

        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(0, batch, filter_depth=self.config['mapping']['filter_depth'])

        self.last_image_pose = batch['c2w'].detach().clone()
        self.last_image_gt_pose = batch['c2w'].detach().clone()
        self.last_image_depth = batch['depth']
        self.last_image_frame_id = batch['frame_id']

        print('First frame mapping done')
        print('last image id = ', self.last_image_frame_id)

        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points - 1) * voxel_size
        offset_max = self.bounding_box[:, 1] - self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1, 1, 1, 3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
        tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
        tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()

        loss = (tv_x + tv_y + tv_z) / (sample_points ** 3)

        return loss

    def get_pose_param_optim(self, poses, frame_id=0, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))

        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                           {"params": cur_trans, "lr": self.config[task]['lr_trans']}])

        return cur_rot, cur_trans, pose_optimizer

    def current_frame_mapping(self, batch, cur_frame_id):
        '''
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        '''
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('Current frame mapping...')

        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.cur_map_optimizer.step()

        return ret, loss

    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack(
            [self.est_c2w_data[i] for i in range(0, self.track_img_num, self.config['mapping']['keyframe_every'])])

        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, self.track_img_num, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[self.track_img_num][None, ...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)

        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[self.track_img_num][None, ...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            # TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),
                                    max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids),
                                        self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0)  # N, 7
            ids_all = torch.cat([ids // self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(
                torch.int64)

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.get_loss_from_ret(ret, smooth=True)

            loss.backward(retain_graph=True)

            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:

                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

                else:
                    current_pose = self.est_c2w_data[self.track_img_num][None, ...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

                # zero_grad here
                pose_optimizer.zero_grad()

        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i + 1].item())] = \
                    self.matrix_from_tensor(cur_rot[i:i + 1], cur_trans[i:i + 1]).detach().clone()[0]

            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[self.track_img_num] = \
                    self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]

    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev.clone()

        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id - 2].clone().to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id - 1].clone().to(self.device)
            delta = c2w_est_prev @ c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta @ c2w_est_prev

        return self.est_c2w_data[frame_id]

    def tracking_render(self, batch, frame_id):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''

        print('tracking image: ', batch['img_path'])

        c2w_gt = batch['c2w'][0].to(self.device)

        mkpts_cur, mkpts_last, cur_feats, mscores, curr_desc_all = self.select_samples_from_sp(self.dataset.H, self.dataset.W,
                                                                                self.config['tracking']['sample'],
                                                                                batch['rgb'], batch)

        if mkpts_cur.shape[0] >= 350:
            if frame_id - self.last_image_frame_id < self.config['data']['run_interval']:
                return True

        self.track_img_num += 1

        # Initialize current pose
        cur_c2w = self.predict_current_pose(self.track_img_num, self.config['tracking']['const_speed'])

        indice = None
        best_sdf_loss = None
        thresh = 0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None, ...], self.track_img_num, mapping=False)

        pose_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pose_optimizer,
                                                                    T_max=self.config['tracking']['iter'],
                                                                    eta_min=1e-3)

        # Start tracking
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

            if iW != 0 and iH != 0:
                map_indice = self.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2,
                                                 self.config['tracking']['sample'], batch['rgb'])
                # Slicing
                indice_h, indice_w = map_indice % (self.dataset.H - iH * 2), map_indice // (self.dataset.H - iH * 2)
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
            else:
                map_indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['tracking']['sample'],
                                                 batch['rgb'])
                #  Slicing
                indice_h, indice_w = map_indice % (self.dataset.H), map_indice // (self.dataset.H)
                rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
                target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
                target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w_est[..., :3, -1].repeat(self.config['tracking']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            if self.config['loss']['use_warp_loss']:
                # # warping loss + rendering loss
                loss = 0.0
                loss_pose, loss_kp_mean = self.get_loss_from_match(mkpts_cur, mkpts_last, c2w_gt, batch, mscores, i)
                loss = loss + loss_pose * self.config['loss']['warp_loss_weight']
                # loss += self.get_loss_from_ret(ret, rgb=False, sdf=True, depth=False, fs=True, smooth=False)
                loss += self.get_loss_from_ret(ret)
            else:
                # # only rendering loss
                loss = self.get_loss_from_ret(ret)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh += 1

            if thresh > self.config['tracking']['wait_iters']:
                break

            loss.backward()
            pose_optimizer.step()
            pose_scheduler.step()

            # # early stop policy
            # if loss_kp_mean < 0.005:
            #     print('error of kp warping is small, no need for more iterations, stop opt pose from iter {}'.format(i))
            #     break

        if self.config['tracking']['best']:
            # Use the pose with smallest loss
            self.est_c2w_data[self.track_img_num] = best_c2w_est.detach().clone()[0]
            self.last_image_pose = best_c2w_est.detach().clone()
        else:
            # Use the pose after the last iteration
            self.est_c2w_data[self.track_img_num] = c2w_est.detach().clone()[0]
            self.last_image_pose = c2w_est.detach().clone()

        self.gt_c2w_data[self.track_img_num] = c2w_gt

        # Save relative pose of non-keyframes
        if self.track_img_num % self.config['mapping']['keyframe_every'] != 0:
            kf_id = self.track_img_num // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[self.track_img_num] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[self.track_img_num] = delta

        self.last_image = batch['rgb']
        self.last_image_depth = batch['depth']
        self.last_image_kp = cur_feats
        self.last_image_feature = curr_desc_all
        self.last_image_gt_pose = batch['c2w'].clone().detach()
        self.last_image_frame_id = batch['frame_id']

        print(
            'Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3]).cpu().item(),
                                                F.l1_loss(c2w_est[0, :3], c2w_gt[:3]).cpu().item()))

        return False

    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i]
                poses[i] = delta @ c2w_key

        return poses

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6,
                                 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15,
                                 'lr': self.config['mapping']['lr_embed']}]

        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15,
                                         'lr': self.config['mapping']['lr_embed_color']})

        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))

    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                     'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf,
                     self.config,
                     self.bounding_box,
                     color_func=color_func,
                     marching_cube_bound=self.marching_cube_bound,
                     voxel_size=voxel_size,
                     mesh_savepath=mesh_savepath)

    def run(self):
        self.create_optimizer()
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        # Start Hero-SLAM!
        for i, batch in tqdm(enumerate(data_loader)):
            # Visualisation
            if self.config['mesh']['visualisation']:
                rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)
                raw_depth = batch["depth"]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                image = np.hstack((rgb, depth_colormap))
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

            # First frame mapping
            if i == 0:
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
                print('after first mapping')
            # Tracking + Mapping
            else:
                skip_image = self.tracking_render(batch, i)
                if skip_image:
                    continue

                if self.track_img_num % self.config['mapping']['map_every'] == 0:
                    print('global BA:', i, self.track_img_num)
                    # self.current_frame_mapping(batch, i)
                    self.global_BA(batch, i)
                    self.last_image_pose = self.est_c2w_data[self.track_img_num].clone().detach()
                    self.last_image_gt_pose = batch['c2w'].clone().detach()

                # Add keyframe
                if self.track_img_num % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframeDatabase.add_keyframe(self.track_img_num, batch, filter_depth=self.config['mapping']['filter_depth'])
                    print('add keyframe:', i)

                if i % 1 == 0:
                    if i % self.config['mesh']['vis'] == 0:
                        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])

                    pose_relative = self.convert_relative_pose()
                    pose_evaluation(self.gt_c2w_data, self.est_c2w_data, 1,
                                    os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, align_or_not=True)
                    pose_evaluation(self.gt_c2w_data, self.est_c2w_data, 1,
                                    os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i,
                                    align_or_not=False, img='pose_abs', name='output_abs.txt')
                    pose_evaluation(self.gt_c2w_data, pose_relative, 1,
                                    os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, align_or_not=True,
                                    img='pose_r', name='output_relative.txt')

                    if cfg['mesh']['visualisation']:
                        cv2.namedWindow('Traj:'.format(i), cv2.WINDOW_AUTOSIZE)
                        traj_image = cv2.imread(
                            os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                         "pose_r_{}.png".format(i)))
                        # best_traj_image = cv2.imread(os.path.join(best_logdir_scene, "pose_r_{}.png".format(i)))
                        # image_show = np.hstack((traj_image, best_traj_image))
                        image_show = traj_image
                        cv2.imshow('Traj:'.format(i), image_show)
                        key = cv2.waitKey(1)

        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                      'checkpoint{}.pt'.format(i))

        self.save_ckpt(model_savepath)
        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])

        pose_relative = self.convert_relative_pose()
        pose_evaluation(self.gt_c2w_data, self.est_c2w_data, 1,
                        os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i,
                        align_or_not=True)
        pose_evaluation(self.gt_c2w_data, self.est_c2w_data, 1,
                        os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i,
                        align_or_not=False, img='pose_abs', name='output_abs.txt')
        pose_evaluation(self.gt_c2w_data, pose_relative, 1,
                        os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i,
                        align_or_not=True,
                        img='pose_r', name='output_relative.txt')


if __name__ == '__main__':

    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the HERO-SLAM.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')

    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'config.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = HeroSLAM(cfg)

    slam.run()
