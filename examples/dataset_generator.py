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
    parser.add_argument('-es', '--elevation_step', type=int, default=10)
    parser.add_argument('-rd', '--radius', nargs=2, type=float, default=[2.0, 4.0])
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
    radius = FLAGS.radius

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

    elevation_delta = 160.0 / FLAGS.elevation_step
    elevation_steps = int(160.0 / elevation_delta)
    
    print(f"Generating {azimuth_steps} views...")
    
    view_image_dict = dict()
    mask_image_dict = dict()

    for i in range(azimuth_steps):
        azimuth = i * azimuth_delta
        for j in range(elevation_steps):
            elevation = -80 + j * elevation_delta  # -80도에서 +80도까지
            
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

            #view -> 뷰 공간 좌표계로 변환된 좌표들이야 나는 이걸 0~1 사이로 정규화된 값으로 바꾸고싶다구
            # 마스크 값이 0.1 이상인 영역만 사용하여 min, max 계산
            valid_mask = mask[..., 0] >= 0.1  # [H, W]
            
            # 유효한 영역의 view 좌표만 추출
            valid_view = view[valid_mask]  # [N, 4]
            
            if len(valid_view) > 0:
                # X, Y, Z 좌표에서 min, max 구하기 (마스크 0.1 이상인 영역만)
                view_min_x = valid_view[:, 0].min()
                view_max_x = valid_view[:, 0].max()
                view_min_y = valid_view[:, 1].min()
                view_max_y = valid_view[:, 1].max()
                view_min_z = valid_view[:, 2].min()
                view_max_z = valid_view[:, 2].max()
            else:
                # 유효한 픽셀이 없는 경우 전체 범위 사용
                view_min_x = view[:, :, 0].min()
                view_max_x = view[:, :, 0].max()
                view_min_y = view[:, :, 1].min()
                view_max_y = view[:, :, 1].max()
                view_min_z = view[:, :, 2].min()
                view_max_z = view[:, :, 2].max()
            
            # 디버깅: min/max 값 출력
            # print(f"  X range: [{view_min_x:.4f}, {view_max_x:.4f}]")
            # print(f"  Y range: [{view_min_y:.4f}, {view_max_y:.4f}]")
            # print(f"  Z range: [{view_min_z:.4f}, {view_max_z:.4f}]")

            # 정규화 함수
            def normalize_channel(channel, min_val, max_val):
                return (channel - min_val) / (max_val - min_val + 1e-8)
            
            # 각 채널 정규화 (마스크 영역만)
            view_norm = np.zeros_like(view)
            
            # 마스크가 있는 영역에서만 정규화 수행
            view_norm[valid_mask, 0] = normalize_channel(view[valid_mask, 0], view_min_x, view_max_x)
            view_norm[valid_mask, 1] = normalize_channel(view[valid_mask, 1], view_min_y, view_max_y)
            view_norm[valid_mask, 2] = normalize_channel(view[valid_mask, 2], view_min_z, view_max_z)
            
            # 마스크가 없는 영역은 이미 0으로 초기화되어 있음
            
            # 디버깅: 정규화 후 실제 값 범위 확인
            # print(f"  Normalized ranges:")
            # valid_view_norm = view_norm[valid_mask]
            # print(f"    X: [{valid_view_norm[:, 0].min():.4f}, {valid_view_norm[:, 0].max():.4f}]")
            # print(f"    Y: [{valid_view_norm[:, 1].min():.4f}, {valid_view_norm[:, 1].max():.4f}]")
            # print(f"    Z: [{valid_view_norm[:, 2].min():.4f}, {valid_view_norm[:, 2].max():.4f}]")

            # 파일 저장 (RGB 채널만 사용)
            view_path = os.path.join(FLAGS.out_dir, f"view_{i:03d}.png")
            mask_path = os.path.join(FLAGS.out_dir, f"mask_{i:03d}.png")

            view_image_dict[f"{view_path}"] = view_norm
            mask_image_dict[f"{mask_path}"] = mask
            
            # 데이터셋 항목 추가
            dataset_item = dict()
            dataset_item["view_path"] = os.path.basename(view_path)
            dataset_item["mask_path"] = os.path.basename(mask_path)
            dataset_item["view_min_x"] = float(view_min_x)
            dataset_item["view_min_y"] = float(view_min_y)
            dataset_item["view_min_z"] = float(view_min_z)
            dataset_item["view_max_x"] = float(view_max_x)
            dataset_item["view_max_y"] = float(view_max_y)
            dataset_item["view_max_z"] = float(view_max_z)
            #mv 행렬 저장
            dataset_item["mv"] = mv.cpu().numpy().tolist()  # mv 행렬 저장
            

            dataset["data"].append(dataset_item)
        print(f"Generated views for azimuth {azimuth} degrees.")
    
    #데이터셋 json 저장
    dataset_json_path = os.path.join(FLAGS.out_dir, "dataset.json")

    # imageio.imwrite(view_path, (view_norm[:, :, :3] * 255).astype(np.uint8))
    # imageio.imwrite(mask_path, (mask[..., 0] * 255).astype(np.uint8))

    #멀티스레드르 이용해서 저장한다.
    import concurrent.futures
    def save_image(path, image, is_mask=False):
        if is_mask:
            imageio.imwrite(path, (image[..., 0] * 255).astype(np.uint8))
        else:
            imageio.imwrite(path, (image[:, :, :3] * 255).astype(np.uint8))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for path, image in view_image_dict.items():
            futures.append(executor.submit(save_image, path, image, is_mask=False))
        for path, image in mask_image_dict.items():
            futures.append(executor.submit(save_image, path, image, is_mask=True))
        #모든 작업이 완료될 때까지 대기
        concurrent.futures.wait(futures)


    #json으로 저장
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Dataset JSON saved to {dataset_json_path}")