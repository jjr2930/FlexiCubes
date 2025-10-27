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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    parser.add_argument('-es', '--elevation_step', type=int, default=10)
    parser.add_argument('-el', '--elevation_limit', nargs=2, type=float, default=[-80, 80])
    parser.add_argument('-rd', '--radius', nargs=2, type=float, default=[2.0, 4.0])
    parser.add_argument('-rr', '--rendering_resolution', nargs=2, type=int, default=[2048,2048])
    parser.add_argument('-lp', '--look_position', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument('-fov', '--fovy', type=float, default=45.0)
    parser.add_argument('-cn', '--cam_near', type=float, default=0.1)
    parser.add_argument('-cf', '--cam_far', type=float, default=100.0)
    parser.add_argument('-fv', '--focus_observe_vertex', nargs=3, type=int, default=[0,1,2])
    parser.add_argument('-fr', '--focus_observe_radius', nargs=2, type=float, default=[2.0,4.0])
    parser.add_argument('-fs', '--focus_observe_count', type=int, default=30)
    
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
    radius = FLAGS.radius
    elevation_limit = FLAGS.elevation_limit
    focus_observe_vertex = FLAGS.focus_observe_vertex #index of vertex to focus and observe
    focus_observe_radius = FLAGS.focus_observe_radius
    focus_observe_count = FLAGS.focus_observe_count

    #print arguments
    print("Arguments:")
    for arg in vars(FLAGS):
        print(f"  {arg}: {getattr(FLAGS, arg)}")

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

    elevation_range = abs(elevation_limit[0]) + abs(elevation_limit[1])
    elevation_delta = elevation_range / FLAGS.elevation_step
    elevation_steps = int(elevation_range / elevation_delta)

    print(f"Generating {azimuth_steps} views...")
    

    for i in range(azimuth_steps):
        azimuth = i * azimuth_delta
        for i in range(elevation_steps):
            elevation = elevation_limit[0] + i * elevation_delta  # -80도에서 +80도까지

            # 한 장씩 렌더링하여 메모리 절약
            random_radius = random.uniform(radius[0], radius[1])

            mv, mvp = render.orbit(elevation=np.deg2rad(elevation), 
                                azimuth=np.deg2rad(azimuth), 
                                radius=random_radius, 
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
            
            view = rendered['depth'][0].cpu().numpy()  # [H, W, 4]
            mask = rendered['mask'][0].cpu().numpy()    # [H, W, 1]
            
            # 메모리 정리
            del rendered
            torch.cuda.empty_cache()

            # 파일 저장
            index = i * elevation_steps + i
            view_path = os.path.join(FLAGS.out_dir, f"view_{index:03d}.npy")
            mask_path = os.path.join(FLAGS.out_dir, f"mask_{index:03d}.png")

            # view는 NumPy 바이너리로 저장 (원본 float32 값 그대로, 정밀도 손실 없음)
            np.save(view_path, view)
            # mask는 PNG로 저장 (단순 0/1 값이므로 PNG로 충분)
            imageio.imwrite(mask_path, (mask[..., 0] * 255).astype(np.uint8))
            
            # 데이터셋 항목 추가
            dataset_item = dict()
            dataset_item["view_path"] = os.path.basename(view_path)
            dataset_item["mask_path"] = os.path.basename(mask_path)
            # mv 행렬 저장
            dataset_item["mv"] = mv.cpu().numpy().tolist()

            dataset["data"].append(dataset_item)
        print(f"Generated views for azimuth {azimuth} degrees.")

    print("Generated focus observe views.")
    
    for i in range(3):
        if i < 0 or i >= len(focus_observe_vertex):
            print(f"Skipping invalid focus observe vertex index: {i}")
            continue

        vertex_index = gt_mesh.faces[focus_observe_vertex[i]][0]  # 해당 삼각형의 0번째 정점 인덱스
        vertex_coord = gt_mesh.vertices[vertex_index]   # 해당 정점의 좌표 (Tensor)
            
        random_radius = random.uniform(focus_observe_radius[0], focus_observe_radius[1])

        mv, mvp = render.get_random_camera_batch_custom(focus_observe_count, 
                                                        fovy=fovy,
                                                        iter_res=render_resolution,
                                                        position=vertex_coord.cpu().numpy(), device=device)

        # get_random_camera_batch_custom은 이미 배치 형태 [focus_observe_count, 4, 4]로 반환하므로 unsqueeze 불필요
        # 렌더링
        rendered = render.render_mesh_paper(gt_mesh, mv, mvp, render_resolution)
        
        # rendered['depth']와 rendered['mask']는 배치 형태 [focus_observe_count, H, W, C]
        views = rendered['depth'].cpu().numpy()  # [B, H, W, 4]
        masks = rendered['mask'].cpu().numpy()   # [B, H, W, 1]
        
        # 메모리 정리
        del rendered
        torch.cuda.empty_cache()

        # 파일 저장 focus_observe_count 만큼 배치로 렌더링된 결과를 하나씩 저장
        for k in range(focus_observe_count):
            index = k * focus_observe_count + i

            view_path = os.path.join(FLAGS.out_dir, f"view_fc{index:04d}.npy")
            mask_path = os.path.join(FLAGS.out_dir, f"mask_fc{index:04d}.png")

            # view는 NumPy 바이너리로 저장 (원본 float32 값 그대로, 정밀도 손실 없음)
            np.save(view_path, views[k])  # k번째 배치 아이템 저장
            # mask는 PNG로 저장 (단순 0/1 값이므로 PNG로 충분)
            imageio.imwrite(mask_path, (masks[k, ..., 0] * 255).astype(np.uint8))  # k번째 배치 아이템 저장
        
            # 데이터셋 항목 추가
            dataset_item = dict()
            dataset_item["view_path"] = os.path.basename(view_path)
            dataset_item["mask_path"] = os.path.basename(mask_path)
            # mv 행렬 저장 (k번째 배치 아이템)
            dataset_item["mv"] = mv[k].cpu().numpy().tolist()
            dataset["data"].append(dataset_item)  

    print("Focus observe views generated.")  

    #데이터셋 json 저장
    dataset_json_path = os.path.join(FLAGS.out_dir, "dataset.json")

    #json으로 저장
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Dataset JSON saved to {dataset_json_path}")