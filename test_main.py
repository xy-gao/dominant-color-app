import os

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.mark.parametrize(
    "image_file",
    [
        ("test_img1.jpeg"),
        ("test_img2.png"),
    ],
)
def test_extract(image_file):
    image_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"dominant_color/tests/test_imgs/{image_file}",
    )
    with open(image_path, "rb") as img:
        response = client.post("/extract/?n_color=5", files={"file": img})

    print(response.json())
    result = response.json()["result"]
    assert response.status_code == 200
    assert len(result) == 5
    for r in result:
        assert isinstance(r["rgb"]["r"], int)
        assert isinstance(r["rgb"]["g"], int)
        assert isinstance(r["rgb"]["b"], int)
        assert isinstance(r["rate"], float)


@pytest.mark.parametrize(
    "number_of_color",
    [
        (0),
        (11),
    ],
)
def test_extract_number_out_of_range(number_of_color):
    image_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dominant_color/tests/test_imgs/test_img1.jpeg",
    )
    with open(image_path, "rb") as img:
        response = client.post(
            f"/extract/?n_color={number_of_color}", files={"file": img}
        )
    assert response.status_code == 422
