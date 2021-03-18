import shutil
from tempfile import NamedTemporaryFile
from typing import List

from fastapi import FastAPI, File, Query, UploadFile
from pydantic import BaseModel, Field

from dominant_color import DominantColorExtractor

app = FastAPI()


class Rgb(BaseModel):
    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)


class ColorRate(BaseModel):
    rgb: Rgb
    rate: float = Field(ge=0, le=1)


class ResponseModel(BaseModel):
    result: List[ColorRate]


@app.post("/extract/", response_model=ResponseModel)
def extract(n_color: int = Query(..., gt=0, le=10), file: UploadFile = File(...)):
    with NamedTemporaryFile() as tmp:
        shutil.copyfileobj(file.file, tmp)
        result = DominantColorExtractor.create(tmp.name).extract(n_color)
    return ResponseModel(
        result=[
            ColorRate(rgb=Rgb(**dict(zip(["r", "g", "b"], rgb.tolist()))), rate=rate)
            for rgb, rate in result
        ]
    )
