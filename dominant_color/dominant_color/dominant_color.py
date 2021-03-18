from collections import Counter, OrderedDict
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


class DominantColorExtractor:
    def __init__(self, image_path: str, image_rgbs: np.ndarray):
        self.image_path = image_path
        self.image_rgbs = image_rgbs

    @staticmethod
    def _read_image_to_rgbs(image_path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _resize(image_array: np.ndarray) -> np.ndarray:
        return cv2.resize(image_array, (256, 256))

    @staticmethod
    def _reshape_to_two_dim(image_array: np.ndarray) -> np.ndarray:
        return image_array.reshape(-1, 3)

    @staticmethod
    def _extract(
        rgb_array: np.ndarray, number_of_colors: int
    ) -> List[Tuple[np.ndarray, float]]:
        clusters = KMeans(n_clusters=number_of_colors).fit(rgb_array)
        total_len = len(clusters.labels_)
        color_count: Sequence[Tuple[int, int]] = Counter(clusters.labels_).most_common()
        return [
            (clusters.cluster_centers_[idx].astype("uint8"), count / total_len)
            for idx, count in color_count
        ]

    @classmethod
    def create(cls, image_path: str):
        return cls(image_path, cls._read_image_to_rgbs(image_path))

    def extract(
        self, number_of_colors: int, resize: bool = True
    ) -> List[Tuple[np.ndarray, float]]:
        if resize:
            rgbs = self._resize(self.image_rgbs)
        else:
            rgbs = self.image_rgbs
        return self._extract(self._reshape_to_two_dim(rgbs), number_of_colors)
