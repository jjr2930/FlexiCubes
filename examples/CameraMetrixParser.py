import json
import os
from typing import List, Dict, Any

import numpy as np


class CameraMetrixParser:
    """
    CameraModelViewMatrix.json 파일을 파싱하여 모델뷰 행렬 리스트를 보관하는 클래스.

    - 원시 데이터: `raw_model_view_list` (List[Dict[str, float]])
    - 4x4 행렬: `model_view_matrices` (List[np.ndarray], dtype=float32, shape=(4,4))
    """

    def __init__(self, json_path: str) -> None:
        self.json_path: str = json_path
        self.raw_model_view_list: List[Dict[str, Any]] = []
        self.model_view_matrices: List[np.ndarray] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if "modelViewMatrix" not in data or not isinstance(data["modelViewMatrix"], list):
            raise ValueError("Invalid JSON format: missing 'modelViewMatrix' list")

        self.raw_model_view_list = data["modelViewMatrix"]
        self.model_view_matrices = [self._to_matrix(entry) for entry in self.raw_model_view_list]

    @staticmethod
    def _to_matrix(entry: Dict[str, Any]) -> np.ndarray:
        """
        e00..e33 키를 갖는 딕셔너리를 4x4 numpy 행렬로 변환.
        키가 누락되면 0.0(또는 대각 1.0)으로 채움.
        """
        def val(key: str, default: float) -> float:
            v = entry.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        m = np.zeros((4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                key = f"e{r}{c}"
                default = 1.0 if r == c else 0.0
                m[r, c] = val(key, default)
        return m

    def count(self) -> int:
        return len(self.model_view_matrices)

    def get_matrix(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.count():
            raise IndexError("matrix index out of range")
        return self.model_view_matrices[index]

    def as_list(self) -> List[np.ndarray]:
        return list(self.model_view_matrices)
    
    def getRandomMatrix(self) -> np.ndarray:
        return np.random.choice(self.model_view_matrices)


if __name__ == "__main__":
    # 간단한 수동 테스트
    json_path = os.path.join(os.path.dirname(__file__), "data", "inputmodels", "CameraModelViewMatrix.json")
    parser = CameraMetrixParser(json_path)
    print(f"Loaded {parser.count()} model-view matrices")
    print(parser.get_matrix(0))


