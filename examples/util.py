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
import numpy as np
import torch
import trimesh
import kaolin
import nvdiffrast.torch as dr

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return x / length(x, eps)

def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)
    
def viewMatrix(eye, at, up, device=None):
    zaxis = safe_normalize(eye - at)
    xaxis = safe_normalize(torch.cross(up, zaxis))
    yaxis = torch.cross(zaxis, xaxis)
    m = torch.eye(4, device=device)
    m[0, :3] = xaxis
    m[1, :3] = yaxis
    m[2, :3] = zaxis
    m[:3, 3] = -torch.matmul(m[:3, :3], eye)
    return m

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        
    def auto_normals(self):
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        nrm = safe_normalize(torch.cross(v1 - v0, v2 - v0))
        self.nrm = nrm

def load_mesh(path, device):
    mesh_np = trimesh.load(path, process=False)
    vertices = torch.tensor(mesh_np.vertices, device=device, dtype=torch.float)    
    faces = torch.tensor(mesh_np.faces, device=device, dtype=torch.long)
    #면 과 정점 갯수 출력
    print(f"Loaded mesh from {path}, #faces: {faces.shape[0]}, #vertices: {vertices.shape[0]}")
    
    
    # Normalize
    vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
    scale = 1.8 / torch.max(vmax - vmin).item()
    vertices = vertices - (vmax + vmin) / 2 # Center mesh on origin
    vertices = vertices * scale # Rescale to [-0.9, 0.9]
    return Mesh(vertices, faces)

def compute_sdf(points, vertices, faces):
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices.clone().unsqueeze(0), faces)
    distance = kaolin.metrics.trianglemesh.point_to_mesh_distance(points.unsqueeze(0), face_vertices)[0]
    with torch.no_grad():
        sign = (kaolin.ops.mesh.check_sign(vertices.unsqueeze(0), faces, points.unsqueeze(0))<1).float() * 2 - 1
    sdf = (sign*distance).squeeze(0)
    return sdf

def sample_random_points(n, mesh):
    pts_random = (torch.rand((n//2,3),device='cuda') - 0.5) * 2
    pts_surface = kaolin.ops.mesh.sample_points(mesh.vertices.unsqueeze(0), mesh.faces, 500)[0].squeeze(0)
    pts_surface += torch.randn_like(pts_surface) * 0.05
    pts = torch.cat([pts_random, pts_surface])
    return pts

def xfm_points(points, matrix):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(
        torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr, rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def ndc_image_to_view_space(ndc_image, depth_image, fovy, resWidth, resHeight, near_clip, far_clip, device=None):
    """
    NDC(정규화된 장치 좌표) 깊이 이미지를 뷰 공간 깊이 이미지로 변환.
    ndc_image: [H, W, r,g,b,a] ndc 공간에서의 3차원 좌표(0~1 범위, -1~1 로 복원해야함)
    depth_image: [H, W, 1] 깊이 이미지(0~1 범위, NDC depth)=> 
    fovy: 수직 시야각 (라디안)
    resWidth: 이미지 너비
    resHeight: 이미지 높이
    near_clip: 근접 클리핑 평면
    far_clip: 원거리 클리핑 평면
    Returns: 뷰 공간 깊이 이미지 [H, W, x,y,z,w]

    구현순서
    -. NDC 이미지는 [0,1] 이므로 [-1, 1] 범위로 변환
    -. depth 이미지도 [0,1] 범위이므로 [-1,1]로 변환
    -. near clip, far clip 이용해 NDC depth를 뷰 공간 깊이로 복원 복원한 값을 z_view 라고 하자.
    -. z_view 를 ndc_coords 에 곱하기
    -. 투영 행렬과 역행렬 계산
    -. 각 픽셀의 NDC 좌표에 역투영 행렬 적용
    -. 복원 된 값 반환
    """
    if device is None:
        device = ndc_image.device

    aspect = resWidth / resHeight

    # NDC 좌표를 [-1, 1] 범위로 변환
    ndc_coords = ndc_image * 2 - 1

    #near clip, far clip 이용해 NDC depth를 뷰 공간 깊이로 복원
    depth_ndc = depth_image * 2 - 1  # [H, W, 1]

    z_view = (2 * far_clip * near_clip) / (far_clip + near_clip - depth_ndc * (far_clip - near_clip))

    #Z_VIEW를 ndc_image에서 rgba에다가 곱하기
    ndc_coords *= z_view

    # 투영 행렬과 역행렬
    perspective_matrix = perspective(fovy, aspect, near_clip, far_clip, device=device)
    inv_perspective_matrix = torch.inverse(perspective_matrix)

    # 각 픽셀의 NDC 좌표에 역투영 행렬 적용
    H, W, _ = ndc_coords.shape
    ndc_coords_flat = ndc_coords.view(-1, 4)  # [H*W, 4]
    view_space_coords_flat = torch.matmul(
        torch.nn.functional.pad(ndc_coords_flat[:, :3], pad=(0, 1), mode='constant', value=1.0),
        inv_perspective_matrix.T
    )  # [H*W, 4]
    view_space_coords = view_space_coords_flat.view(H, W, 4)  # [H, W, 4]

    return view_space_coords

def recorver_view_position(normalized_view, mask, min_x, max_x, min_y, max_y, min_z, max_z):
    """
    정규화된 뷰 공간 좌표를 원래 뷰 공간 좌표로 복원
    normalized_view: [H, W, x,y,z,w] 0~1 사이로 정규화된 뷰 공간 좌표
    mask: [H, W, 1] 마스크 이미지
    min_x, max_x, min_y, max_y, min_z, max_z: 각 축의 최소/최대 값
    Returns: 복원된 뷰 공간 좌표 [H, W, x,y,z,w]
    """
    view = torch.zeros_like(normalized_view)

    # 마스크가 있는 영역에서만 복원 수행
    valid_mask = mask[..., 0] >= 0.1  # [H, W]

    # X, Y, Z 좌표 복원 (마스크 영역만)
    view[valid_mask, 0] = normalized_view[valid_mask, 0] * (max_x - min_x) + min_x
    view[valid_mask, 1] = normalized_view[valid_mask, 1] * (max_y - min_y) + min_y
    view[valid_mask, 2] = normalized_view[valid_mask, 2] * (max_z - min_z) + min_z

    return view