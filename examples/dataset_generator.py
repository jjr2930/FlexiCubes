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
    parser.add_argument('-as', '--azimth_step', type=int, default=10)
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

    mvstack, mvpstack = [], []
    azimuth_delta = 360.0/FLAGS.azimuth_step
    for azimuth in range(0.0, 360.0, azimuth_delta):
        mv, mvp = render.orbit(0, azimuth=azimuth, radius=radius, 
                            lookPosition=lookPosition, 
                            fovy=np.deg2rad(fovy), 
                            iter_res=render_resolution, 
                            cam_near_far=[cam_near, cam_far],
                            device=device)
    
        mvstack.append(mv)
        mvpstack.append(mvp)

    mv = torch.stack(mvstack).to(device)
    mvp = torch.stack(mvpstack).to(device)
    
    rendered = render.render_mesh_paper(gt_mesh, mv, mvp, render_resolution)
    
    depth = rendered['depth'].cpu().numpy()
    mask = rendered['mask'].cpu().numpy()

    #depth, 모두 0~1사이로 정규화
    depth_min = depth[mask>0].min()
    depth_max = depth[mask>0].max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    
    batch = mv.shape[0]
    dataset = dict()
    dataset["fovy"] = fovy
    dataset["near"] = cam_near
    dataset["far"] = cam_far
    dataset["res_width"] = render_resolution[0]
    dataset["res_height"] = render_resolution[1]
    dataset["data"] = []

    for i in range(batch):
        depth_path = os.path.join(FLAGS.out_dir, f"depth_{i:03d}.png")
        mask_path = os.path.join(FLAGS.out_dir, f"mask_{i:03d}.png")

        imageio.imwrite(depth_path, (depth_norm[i]*255).astype(np.uint8))
        imageio.imwrite(mask_path, (mask[i]*255).astype(np.uint8))
        print("Depth and mask images saved.")

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