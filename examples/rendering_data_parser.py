"""카메라 매트릭스 JSON 파서.

첨부된 Unity 출력(JSON)을 읽어 RGB/Depth/Mask 경로 및 4x4 카메라 변환 행렬을
파이썬 객체로 변환한다.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import torch
import util
from PIL import Image
import render

@dataclass(frozen=True)
class CameraItem:
	"""단일 프레임의 메타데이터와 카메라 행렬 정보를 보관."""

	rgb_path: str
	depth_path: str
	mask_path: str
	matrix: List[List[float]]

	def to_torch_matrix(self, device=None) -> 'torch.Tensor':
		"""4x4 행렬을 torch.Tensor로 변환."""
		return torch.tensor(self.matrix, dtype=torch.float32, device=device)

	def load_rgb_image(self, working_directory: str) -> 'torch.Tensor':
		"""RGB 이미지를 로드해 [H, W, 3] torch.Tensor로 반환."""	
		fullPath = Path(working_directory) / self.rgb_path
		img = Image.open(fullPath).convert("RGB")
		return torch.from_numpy(np.array(img)).float() / 255.0
	
	def load_depth_image(self, working_directory: str) -> 'torch.Tensor':
		"""Depth 이미지를 로드해 [H, W] torch.Tensor로 반환."""	
		fullPath = Path(working_directory) / self.depth_path
		img = Image.open(fullPath).convert("L")
		return torch.from_numpy(np.array(img)).float() / 255.0

	def load_mask_image(self, working_directory: str) -> 'torch.Tensor':
		"""Mask 이미지를 로드해 [H, W] torch.Tensor로 반환."""	
		fullPath = Path(working_directory) / self.mask_path
		img = Image.open(fullPath).convert("L")
		return torch.from_numpy(np.array(img)).float() / 255.0
	
	def to_pipeline_dict(self, working_directory: str):
		return render.render_from_images(self.load_rgb_image(working_directory=working_directory), 
								   self.load_mask_image(working_directory=working_directory), 
								   self.load_depth_image(working_directory=working_directory))


@dataclass(frozen=True)
class RenderingDataSet:
	"""카메라 메타데이터 셋."""

	fov: float
	near_clip: float
	far_clip: float
	resolution_width: int
	resolution_height: int

	items: List[CameraItem]

	def get_random_items(self, item_count: int) -> List[CameraItem]:
		"""랜덤하게 선택된 CameraItem 리스트를 반환.
		
		Args:
			item_count: 선택할 아이템 개수
			
		Returns:
			랜덤하게 선택된 CameraItem 리스트 (중복 없음)
		"""
		import random
		# item_count가 전체 아이템 수보다 크면 전체 아이템 수로 제한
		actual_count = min(item_count, len(self.items))
		return random.sample(self.items, actual_count)

	def get_torch_projection_matrix(self, device=None) -> 'torch.Tensor':
		"""투영 행렬을 torch.Tensor로 변환."""
		return util.perspective(self.fov, self.resolution_width / self.resolution_height, self.near_clip, self.far_clip, device=device)
		

def _matrix_dict_to_rows(matrix_dict: dict) -> List[List[float]]:
	"""e00~e33 형태의 평면 딕셔너리를 4x4 행렬(행 우선)로 변환."""

	rows: List[List[float]] = []
	for row in range(4):
		prefix = f"e{row}"
		row_values = [matrix_dict.get(f"{prefix}{col}") for col in range(4)]
		if any(value is None for value in row_values):
			missing = [f"{prefix}{col}" for col, value in enumerate(row_values) if value is None]
			raise KeyError(f"행렬 항목이 누락되었습니다: {', '.join(missing)}")
		rows.append(row_values)
	return rows


def load_camera_dataset(json_path: Path | str) -> RenderingDataSet:
	"""camera_matrices.json 파일을 로드해 `CameraDataset`으로 변환."""

	path = Path(json_path)
	with path.open("r", encoding="utf-8") as handle:
		data = json.load(handle)

	try:
		fov = float(data["fov"])
		near_clip = float(data["nearClip"])
		far_clip = float(data["farClip"])
		resolution_width = int(data["resolutionWidth"])
		resolution_height = int(data["resolutionHeight"])
	except KeyError as exc:  # pragma: no cover - 보호용
		raise KeyError(f"필수 키가 없습니다: {exc.args[0]}") from exc

	items_data = data.get("items", [])
	if not isinstance(items_data, Sequence):
		raise TypeError("'items' 키는 배열이어야 합니다.")

	items: List[CameraItem] = []
	for idx, item in enumerate(items_data):
		try:
			matrix_dict = item["matrix"]
		except KeyError as exc:
			raise KeyError(f"items[{idx}]에 'matrix' 키가 없습니다.") from exc

		matrix_rows = _matrix_dict_to_rows(matrix_dict)
		camera_item = CameraItem(
			rgb_path=str(item.get("rgbPath", "")),
			depth_path=str(item.get("depthPath", "")),
			mask_path=str(item.get("maskPath", "")),
			matrix=matrix_rows,
		)
		items.append(camera_item)

	return RenderingDataSet(fov=fov, 
						near_clip=near_clip, 
						far_clip=far_clip, 
						items=items,
						resolution_width=resolution_width, 
						resolution_height=resolution_height)


def iter_camera_dataset(json_path: Path | str) -> Iterable[CameraItem]:
	"""일회성 전체 로드를 피하고 싶을 때 사용할 수 있는 generator."""

	dataset = load_camera_dataset(json_path)
	return dataset.items


if __name__ == "__main__":
	dataset_path = Path(__file__).with_name("camera_matrices.json")
	dataset = load_camera_dataset(dataset_path)
	print(f"총 {len(dataset.items)}개의 카메라 항목을 로드했습니다. fov={dataset.fov}")
