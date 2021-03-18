import os
from unittest.mock import Mock

import numpy as np
import pytest
from dominant_color import DominantColorExtractor


def test_create(monkeypatch):
    monkeypatch.setattr(DominantColorExtractor, "_read_image_to_rgbs", Mock())
    image_path = "path/to/image"

    actual = DominantColorExtractor.create(image_path)

    assert isinstance(actual, DominantColorExtractor)
    assert actual.image_path == image_path
    assert actual.image_rgbs == DominantColorExtractor._read_image_to_rgbs.return_value
    DominantColorExtractor._read_image_to_rgbs.assert_called_once_with(image_path)


@pytest.mark.parametrize(
    "image_file, expected_shape",
    [
        ("test_imgs/test_img1.jpeg", (933, 1400, 3)),
        ("test_imgs/test_img2.png", (1080, 1920, 3)),
    ],
)
def test__read_image_to_rgbs(image_file, expected_shape):
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_file)

    actual = DominantColorExtractor._read_image_to_rgbs(image_path)
    assert actual.shape == expected_shape


@pytest.fixture
def patch_methods(monkeypatch):
    monkeypatch.setattr(DominantColorExtractor, "_resize", Mock())
    monkeypatch.setattr(DominantColorExtractor, "_reshape_to_two_dim", Mock())
    monkeypatch.setattr(DominantColorExtractor, "_extract", Mock())


def test_extract_resize(patch_methods):
    number_of_colors = Mock(spec=int)
    image_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_imgs/test_img1.jpeg"
    )

    sut = DominantColorExtractor.create(image_path)
    actual = sut.extract(number_of_colors)

    DominantColorExtractor._resize.assert_called_once_with(sut.image_rgbs)
    DominantColorExtractor._reshape_to_two_dim.assert_called_once_with(
        DominantColorExtractor._resize.return_value
    )
    DominantColorExtractor._extract.assert_called_once_with(
        DominantColorExtractor._reshape_to_two_dim.return_value, number_of_colors
    )
    assert actual == DominantColorExtractor._extract.return_value


def test_extract_original_size(patch_methods):
    number_of_colors = Mock(spec=int)
    image_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_imgs/test_img1.jpeg"
    )

    sut = DominantColorExtractor.create(image_path)
    actual = sut.extract(number_of_colors, resize=False)

    DominantColorExtractor._reshape_to_two_dim.assert_called_once_with(sut.image_rgbs)
    DominantColorExtractor._extract.assert_called_once_with(
        DominantColorExtractor._reshape_to_two_dim.return_value, number_of_colors
    )
    assert actual == DominantColorExtractor._extract.return_value


def test__extract():
    number_of_colors = 5
    rgb_array = np.random.rand(10, 3)

    actual = DominantColorExtractor._extract(rgb_array, number_of_colors)

    assert len(actual) == 5
    for color, ratio in actual:
        assert color.shape == (3,)
        assert color.dtype == "uint8"
        assert ratio >= 0
        assert ratio <= 1


def test__reshape_to_two_dim():
    image_array = np.random.rand(10, 5, 3)

    actual = DominantColorExtractor._reshape_to_two_dim(image_array)

    assert actual.shape == (50, 3)


def test__resize():
    image_array = np.random.rand(1000, 1200, 3)

    acutal = DominantColorExtractor._resize(image_array)

    assert acutal.shape == (256, 256, 3)
