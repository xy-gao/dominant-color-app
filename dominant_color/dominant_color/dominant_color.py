from typing import List

import cv2
import numpy as np


class DominantColorExtractor:
    def __init__(self, image_path: str, image_rgbs: np.ndarray):
        self.image_path = image_path
        self.image_rgbs = image_rgbs

    @staticmethod
    def _read_image_to_rgbs(image_path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _resize(image_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _reshape_to_two_dim(image_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _extract(number_of_colors: int, rgb_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def create(cls, image_path: str):
        return cls(image_path, cls._read_image_to_rgbs(image_path))

    def extract(self, number_of_colors: int, resize: bool = True) -> np.ndarray:
        if resize:
            rgbs = self._resize(self.image_rgbs)
        else:
            rgbs = self.image_rgbs
        return self._extract(number_of_colors, self._reshape_to_two_dim(rgbs))
