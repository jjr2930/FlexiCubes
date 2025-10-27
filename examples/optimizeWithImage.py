# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
import os
import time
from util import *
import render
import loss
import imageio.v2 as imageio
from datetime import datetime, timezone, timedelta
import torch.nn.functional as F

import sys
sys.path.append('..')
from flexicubes import FlexiCubes
import random

import json

# Robust boolean parser for argparse (supports: true/false, yes/no, 1/0, and flag-only)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    val = str(v).lower()
    if val in ('yes', 'true', 't', 'y', '1'):
        return True
    if val in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected (true/false).')

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

def time_to_string(seconds, prefix=''):
    seconds = int(seconds)
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{prefix} | {h:02d}:{m:02d}:{s:02d}"

def save_target_images(target, iteration, output_dir):
    """Save target mask and depth images to files"""
    target_images_dir = os.path.join(output_dir, 'target_images')
    os.makedirs(target_images_dir, exist_ok=True)
    
    # Save mask images
    mask_images = target['mask'].detach().cpu().numpy()
    for i in range(mask_images.shape[0]):
        mask_img = (mask_images[i] * 255).astype(np.uint8)
        # Convert single channel to 3-channel RGB image
        if len(mask_img.shape) == 3 and mask_img.shape[-1] == 1:
            mask_img = np.repeat(mask_img, 3, axis=-1)
        elif len(mask_img.shape) == 2:
            mask_img = np.stack([mask_img, mask_img, mask_img], axis=-1)
        imageio.imwrite(os.path.join(target_images_dir, f'mask_iter_{iteration:04d}_batch_{i:02d}.png'), mask_img)
    
    # Save depth images
    depth_images = target['depth'].detach().cpu().numpy()
    for i in range(depth_images.shape[0]):
        # Normalize depth values to 0-255 range
        depth_img = depth_images[i]
        depth_min, depth_max = depth_img.min(), depth_img.max()
        if depth_max > depth_min:
            depth_img = (depth_img - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_img = np.zeros_like(depth_img)
        depth_img = depth_img.astype(np.uint8)
        # Convert single channel to 3-channel RGB image
        if len(depth_img.shape) == 3 and depth_img.shape[-1] == 1:
            depth_img = np.repeat(depth_img, 3, axis=-1)
        elif len(depth_img.shape) == 2:
            depth_img = np.stack([depth_img, depth_img, depth_img], axis=-1)
        imageio.imwrite(os.path.join(target_images_dir, f'depth_iter_{iteration:04d}_batch_{i:02d}.png'), depth_img)    

def as_res_vec(res, device):
    if isinstance(res, int):
        return torch.tensor([res, res, res], dtype=torch.float32, device=device)
    return torch.tensor(res, dtype=torch.float32, device=device)

def print_now_time():
    # utc time +9 (korea) 
    nowDate = datetime.now(timezone.utc) + timedelta(hours=9)
    print(f"Now time: {nowDate.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='flexicubes optimization')
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-i', '--iter', type=int, default=1000)
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('-r', '--train_res', nargs=2, type=int, default=[2048, 2048])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-rm', '--ref_mesh', type=str, default=None)
    parser.add_argument('--voxel_grid_res',nargs=3, type=int, default=[64,64,64])
    
    parser.add_argument('--sdf_loss', type=bool, default=False)  # 이미지 기반 복원에서는 GT 메시가 없으므로 False
    parser.add_argument('--develop_reg', type=bool, default=False)
    parser.add_argument('--sdf_regularizer', type=float, default=0.2)
    
    parser.add_argument('-dr', '--display_res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-si', '--save_interval', type=int, default=20)
    parser.add_argument('-ss', '--save_step', type=bool, default=False)
    parser.add_argument('-fc', '--focus_count', type=int, default= 1 )
    parser.add_argument('-wd', '--working_directory', type=str, default=None)
    parser.add_argument('-op', '--output_prefix', type=str, default=None)
    parser.add_argument('-df', '--dataset_file', type=str, default=None)
    parser.add_argument('-fuf', '--focus_using_flag', nargs=3, type=bool, default=[True, True, True])

    # Use robust boolean parsing so both "--print_loss" and "--print_loss false" work as expected
    # parser.add_argument('-pl', '--print_loss', type=str2bool, nargs='?', const=True, default=False,
    #                     help='Print GT vs rendered losses (use --print_loss or --print_loss true to enable)')

    FLAGS = parser.parse_args()
    device = 'cuda'

    dataset_file = FLAGS.dataset_file
    if dataset_file is None:
       raise ValueError("Dataset file must be specified.")

    json_doc = json.load(open(dataset_file, 'r'))

    fov = json_doc['fovy']
    near_clip = json_doc['near']
    far_clip = json_doc['far']
    res_width = json_doc['res_width']
    res_height = json_doc['res_height']
    data = json_doc['data']
    focus_data = [json_doc['focus_data_0'], json_doc['focus_data_1'], json_doc['focus_data_2']]
    focus_using_flag = FLAGS.focus_using_flag

    # 이미지를 미리 로드하지 않고, 필요할 때마다 로드하는 함수 정의
    def load_and_process_image(item):
        #이 함수의 동작 시간을 측정하고 싶다.
        start_time = time.time()
        """매번 호출될 때마다 이미지를 로드하고 전처리"""
        view_full_path = os.path.join(FLAGS.working_directory, item['view_path'])
        mask_full_path = os.path.join(FLAGS.working_directory, item['mask_path'])

        # view는 NumPy 바이너리 파일로 로드 (.npy) - 렌더러의 원본 [H, W, 4] 값을 그대로 사용
        view_img = np.load(view_full_path)
        # mask는 PNG 파일로 로드
        mask_img = imageio.imread(mask_full_path)
        mask_img = mask_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        # view는 좌표/깊이용 4채널 텐서이므로 어떠한 채널 수정도 하지 않는다 (W 채널 보존)
        
        # 마스크를 단일 채널로 변환 [H, W, 1]
        if mask_img.ndim == 3:
            # RGB/RGBA인 경우 첫 번째 채널만 사용
            mask_img = mask_img[..., :1]
        elif mask_img.ndim == 2:
            # 그레이스케일 [H, W]인 경우 채널 차원 추가
            mask_img = mask_img[..., np.newaxis]

        end_time = time.time()
        # print(f"Image loading and processing time: {end_time - start_time:.4f} seconds")

        return view_img, mask_img

    print(f"Dataset contains {len(data)} images. Images will be loaded on-demand to save memory.")

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()
    
    # Load GT mesh
    gt_mesh = load_mesh(FLAGS.ref_mesh, device)
    gt_mesh.auto_normals() # compute face normals for visualization
    
    # ==============================================================================================
    #  Create and initialize FlexiCubes
    # ==============================================================================================
    fc = FlexiCubes(device)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(FLAGS.voxel_grid_res)
    x_nx3 *= 2 # scale up the grid so that it's larger than the target object
    
    sdf = torch.rand_like(x_nx3[:,0]) - 0.1 # randomly init SDF
    sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    # set per-cube learnable weights to zeros
    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
    weight    = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)
    
    #  Retrieve all the edges of the voxel grid; these edges will be utilized to 
    #  compute the regularization loss in subsequent steps of the process.    
    all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
    grid_edges = torch.unique(all_edges, dim=0)
    
    # ==============================================================================================
    #  Setup optimizer
    # ==============================================================================================
    # Learnable global transform parameters
    optimizer = torch.optim.Adam([sdf, weight,deform], lr=FLAGS.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x)) 

    # ==============================================================================================
    #  print now time
    # ==============================================================================================   
    
    start_time = time.time()
    print_now_time()
    
    # ==============================================================================================
    #  Train loop
    # ==============================================================================================   
    data_index = 0;
    for it in range(FLAGS.iter): 
        optimizer.zero_grad()

        proj = perspective(fovy=np.deg2rad(fov), aspect=res_width/res_height, n=near_clip, f=far_clip, device=device)
        mv_batch = []
        mvp_batch = []
        # mvp will be recomputed after mv fix and model composition
        target = []
        #batch count - focus_count 만큼 data에서 랜덤 선택
        selected_data = random.sample(data, FLAGS.batch - (FLAGS.focus_count * 3))
        
        #집중 관찰 뷰
        if focus_using_flag[0]:
            focus_sample_0 = random.sample(focus_data[0], FLAGS.focus_count)
            selected_data.extend(focus_sample_0)
        if focus_using_flag[1]:
            focus_sample_1 = random.sample(focus_data[1], FLAGS.focus_count)
            selected_data.extend(focus_sample_1)
        if focus_using_flag[2]:
            focus_sample_2 = random.sample(focus_data[2], FLAGS.focus_count)
            selected_data.extend(focus_sample_2)
        
        for item in selected_data:
            #print(time_to_string(time.time(), prefix=f"Processing {item['view_path']}"))
            read_mv = item['mv']
            mv = torch.tensor(read_mv, dtype=torch.float32, device=device)
            mvp = proj @ mv

            # 매번 이미지를 로드 (메모리 절약)
            view_img, mask_img = load_and_process_image(item)
            
            # Convert to torch tensors (원본 view 데이터를 그대로 사용, 정규화 복원 불필요)
            view_torch = torch.from_numpy(view_img).float().to(device)  # [H, W, 4]
            mask_torch = torch.from_numpy(mask_img).float().to(device)  # [H, W, 1]

            mv_batch.append(mv)
            mvp_batch.append(mvp)
            target.append({
                'mask': mask_torch,  # [H, W, 1]
                'depth': view_torch,  # [H, W, 4] - 원본 view 공간 좌표
            })

        mv_stack = torch.stack(mv_batch).to(device)  # [B, 4, 4]
        mvp_stack = torch.stack(mvp_batch).to(device)  # [B, 4, 4]

        #print(time_to_string(time.time(), prefix="Before stacking targets"))

        # Stack target tensors from list of dicts to dict of tensors
        target_stacked = {
            'mask': torch.stack([t['mask'] for t in target]),  # [B, H, W, 1]
            'depth': torch.stack([t['depth'] for t in target])  # [B, H, W, 4]
        }

       
        # extract and render FlexiCubes mesh
        voxel_res = as_res_vec(FLAGS.voxel_grid_res, device)
        grid_verts = x_nx3 + (2-1e-8) / (voxel_res * 2) * torch.tanh(deform)
        vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, FLAGS.voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
            gamma_f=weight[:,20], training=True)
        flexicubes_mesh = Mesh(vertices, faces)
        buffers = render.render_mesh_paper(flexicubes_mesh, mv_stack, mvp_stack, FLAGS.train_res)
        
        # evaluate reconstruction loss
        mask_loss = (buffers['mask'] - target_stacked['mask']).abs().mean()
        depth_loss = (((((buffers['depth'] - (target_stacked['depth']))* target_stacked['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
    
        t_iter = it / FLAGS.iter
        sdf_weight = FLAGS.sdf_regularizer - (FLAGS.sdf_regularizer - FLAGS.sdf_regularizer/20)*min(1.0, 4.0 * t_iter)
        reg_loss = loss.sdf_reg_loss(sdf, grid_edges).mean() * sdf_weight # Loss to eliminate internal floaters that are not visible
        reg_loss += L_dev.mean() * 0.5
        reg_loss += (weight[:,:20]).abs().mean() * 0.1
        total_loss = mask_loss + depth_loss + reg_loss

        
        # Log the differences
        # if(FLAGS.print_loss == True):
        #     gt_target = render.render_mesh_paper(gt_mesh, mv_stack, mvp_stack, FLAGS.train_res)
        
        #     mask_loss_with_gt = (buffers['mask'] - gt_target['mask']).abs().mean()
        #     depth_loss_with_gt = (((((buffers['depth'] - (gt_target['depth']))* gt_target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10

        #     diff_mask_loss = mask_loss_with_gt - mask_loss
        #     diff_depth_loss = depth_loss_with_gt - depth_loss

        #     print(f"============================================")
        #     print(f"gt mask loss: {mask_loss_with_gt.item()} vs rendered mask loss: {mask_loss.item()}")
        #     print(f"gt depth loss: {depth_loss_with_gt.item()} vs rendered depth loss: {depth_loss.item()}")
        #     print(f"diff  in mask loss: {diff_mask_loss.item()}, diff in depth loss: {diff_depth_loss.item()}")
        #     print(f"============================================")
        #print(time_to_string(time.time(), prefix="After computing losses"));

        # if FLAGS.sdf_loss: # optionally add SDF loss to eliminate internal structures
        #     with torch.no_grad():
        #         pts = sample_random_points(1000, gt_mesh)
        #         gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
        #     pred_sdf = compute_sdf(pts, flexicubes_mesh.vertices, flexicubes_mesh.faces)
        #     total_loss += torch.nn.functional.mse_loss(pred_sdf, gt_sdf) * 2e3
        
        # optionally add developability regularizer, as described in paper section 5.2
        # if FLAGS.develop_reg:
        #     reg_weight = max(0, t_iter - 0.8) * 5
        #     if reg_weight > 0: # only applied after shape converges
        #         reg_loss = loss.mesh_developable_reg(flexicubes_mesh).mean() * 10
        #         reg_loss += (deform).abs().mean()
        #         reg_loss += (weight[:,:20]).abs().mean()
        #         total_loss = mask_loss + depth_loss + reg_loss 
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()        
        
        if (it % FLAGS.save_interval == 0 or it == (FLAGS.iter-1)): # save normal image for visualization
            with torch.no_grad():
                # extract mesh with training=False
                vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, FLAGS.voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
                gamma_f=weight[:,20], training=False)
                flexicubes_mesh = Mesh(vertices, faces)
                
                flexicubes_mesh.auto_normals() # compute face normals for visualization
                mv, mvp = render.get_rotate_camera(it//FLAGS.save_interval, iter_res=FLAGS.display_res, device=device,use_kaolin=False)
                val_buffers = render.render_mesh_paper(flexicubes_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res, return_types=["normal"], white_bg=True)
                val_image = ((val_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                
                # gt_buffers = render.render_mesh_paper(gt_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res, return_types=["normal"], white_bg=True)
                # gt_image = ((gt_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                # imageio.imwrite(os.path.join(FLAGS.out_dir, '{:04d}.png'.format(it)), np.concatenate([val_image, gt_image], 1))
                imageio.imwrite(os.path.join(FLAGS.out_dir, '{:04d}.png'.format(it)), val_image)
                print(f"Optimization Step [{it}/{FLAGS.iter}], Loss: {total_loss.item():.4f}")
            
    # ==============================================================================================
    #  print now time and duration time
    # ==============================================================================================     
    print_now_time();
    print(f"duration : {time.time() - start_time}")

    # ==============================================================================================
    #  Save ouput
    # ==============================================================================================     
    mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    mesh_np.export(os.path.join(FLAGS.out_dir, f'{FLAGS.output_prefix} | gird_res {FLAGS.voxel_grid_res} | iter {FLAGS.iter} | lr {FLAGS.learning_rate} | fc {FLAGS.focus_count} | train_res {FLAGS.train_res}.obj'))

