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
import imageio
from datetime import datetime, timezone, timedelta

import sys
sys.path.append('..')
from flexicubes import FlexiCubes
import random
import json

def print_now_time():
    # utc time +9 (korea) 
    nowDate = datetime.now(timezone.utc) + timedelta(hours=9)
    print(f"Now time: {nowDate.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='flexicubes optimization')
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)    
    parser.add_argument('-as', '--azimuth_step', type=int, default=10)
    parser.add_argument('-rd', '--radius', type=float, default=2.0)
    parser.add_argument('-rr', '--rendering_resolution', nargs=2, type=int, default=[2048,2048])
    parser.add_argument('-lp', '--look_position', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument('-fov', '--fovy', type=float, default=45.0)
    parser.add_argument('-cn', '--cam_near', type=float, default=0.1)
    parser.add_argument('-cf', '--cam_far', type=float, default=100.0)
    FLAGS = parser.parse_args()

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    
    device = 'cuda'
    
    # Load GT mesh
    gt_mesh = load_mesh(FLAGS.ref_mesh, device)
    gt_mesh.auto_normals() # compute face normals for visualization

    radius = FLAGS.radius
    lookPosition = FLAGS.look_position
    render_resolution = FLAGS.rendering_resolution
    fovy = FLAGS.fovy
    cam_near = FLAGS.cam_near
    cam_far = FLAGS.cam_far

    # 데이터셋 정보 초기화
    dataset = dict()
    dataset["fovy"] = fovy
    dataset["near"] = cam_near
    dataset["far"] = cam_far
    dataset["res_width"] = render_resolution[0]
    dataset["res_height"] = render_resolution[1]
    dataset["data"] = []

    azimuth_delta = 360.0 / FLAGS.azimuth_step
    azimuth_steps = int(360.0 / azimuth_delta)
    
    print(f"Generating {azimuth_steps} views...")
    
    for i in range(azimuth_steps):
        azimuth = i * azimuth_delta
        print(f"Rendering view {i+1}/{azimuth_steps} (azimuth: {azimuth:.1f}°)...")
        
        # 한 장씩 렌더링하여 메모리 절약
        mv, mvp = render.orbit(0, azimuth=np.deg2rad(azimuth), radius=radius, 
                            lookPosition=lookPosition, 
                            fovy=np.deg2rad(fovy), 
                            iter_res=render_resolution, 
                            cam_near_far=[cam_near, cam_far],
                            device=device)
        
        # 배치 차원 추가 (render_mesh_paper는 배치를 기대함)
        mv_batch = mv.unsqueeze(0)
        mvp_batch = mvp.unsqueeze(0)
        
        # 렌더링
        rendered = render.render_mesh_paper(gt_mesh, mv_batch, mvp_batch, render_resolution)
        
        depth = rendered['depth'][0].cpu().numpy()  # [H, W, 4]
        mask = rendered['mask'][0].cpu().numpy()    # [H, W, 1]
        
        # 메모리 정리
        del rendered
        torch.cuda.empty_cache()
        
        # 각 이미지의 마스크 영역에서 depth 값의 min/max 계산
        current_mask = mask[..., 0] > 0  # [H, W]
        
        # 마스크 영역의 depth 좌표들
        masked_depth = depth[current_mask]  # [N, 4]
        
        if len(masked_depth) > 0:
            # 각 채널(x, y, z)의 min/max 계산
            depth_min = masked_depth[:, :3].min(axis=0)  # [3] (x, y, z)
            depth_max = masked_depth[:, :3].max(axis=0)  # [3] (x, y, z)
            
            # depth를 0~1로 정규화 (Z 채널만 사용)
            depth_z = depth[..., 2]  # [H, W]
            depth_z_min = masked_depth[:, 2].min()
            depth_z_max = masked_depth[:, 2].max()
            depth_norm = (depth_z - depth_z_min) / (depth_z_max - depth_z_min + 1e-8)
        else:
            # 마스크가 비어있는 경우 기본값
            depth_min = np.array([-1.0, -1.0, -1.0])
            depth_max = np.array([-1.0, -1.0, -1.0])
            depth_norm = depth[..., 2]
        
        # 파일 저장
        depth_path = os.path.join(FLAGS.out_dir, f"depth_{i:03d}.png")
        mask_path = os.path.join(FLAGS.out_dir, f"mask_{i:03d}.png")

        imageio.imwrite(depth_path, (depth_norm * 255).astype(np.uint8))
        imageio.imwrite(mask_path, (mask[..., 0] * 255).astype(np.uint8))
        
        # 데이터셋 항목 추가
        dataset_item = dict()
        dataset_item["depth_path"] = os.path.basename(depth_path)
        dataset_item["mask_path"] = os.path.basename(mask_path)
        dataset_item["depth_min_x"] = float(depth_min[0])
        dataset_item["depth_min_y"] = float(depth_min[1])
        dataset_item["depth_min_z"] = float(depth_min[2])
        dataset_item["depth_max_x"] = float(depth_max[0])
        dataset_item["depth_max_y"] = float(depth_max[1])
        dataset_item["depth_max_z"] = float(depth_max[2])

        dataset["data"].append(dataset_item)
    
    #데이터셋 json 저장
    dataset_json_path = os.path.join(FLAGS.out_dir, "dataset.json")

    #json으로 저장
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Dataset JSON saved to {dataset_json_path}")