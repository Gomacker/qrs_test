import abc
import asyncio
import dataclasses
from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray

from base.components.onnx_infer import OnnxRun
from base.components.rec import Rec
from base.components.utils.face import getFaceBoxs
# import pydantic

from base.components.utils.ocr import DetRecFunctions, draw_ocr


DET_PATH = Path('./resource/model_zoo/det_model.onnx')
SMALL_REC_PATH = Path('./resource/model_zoo/rec_model.onnx')
LARGE_REC_PATH = Path('./resource/model_zoo/rec_model.onnx')
PPOCR_KEYS_PATH = Path('./resource/model_zoo/ppocr_keys_v1.txt')


class OcrRec(Rec):
    @dataclasses.dataclass
    class Precondition:
        box: ndarray
        text: str
        score: float

    def __init__(self, det_file: Path, small_rec_file: Path, large_rec_file: Path, ppocr_keys: Path):
        self.ocr_sys: DetRecFunctions = DetRecFunctions(
            use_large=False,
            det_file=str(det_file),
            small_rec_file=str(small_rec_file),
            large_rec_file=str(large_rec_file),
            ppocr_keys=str(ppocr_keys)
        )
        self.predictions = []

    async def inference(self, img: ndarray):
        filter_prediction: list = []
        dt_boxes: list = self.ocr_sys.get_boxes(img)

        results, results_info = self.ocr_sys.recognition_img(img, dt_boxes)

        for box, rec_result in zip(dt_boxes, results):

            text, score = rec_result
            if score >= 0.5:
                filter_prediction.append(OcrRec.Precondition(box, text, score))

        self.predictions = filter_prediction
        return self.predictions

    def display(self, img: ndarray, predictions: list, font_path=Path('../../resource/font/simfang.ttf')):
        results = []
        if predictions:
            dt_boxes = [x.box for x in predictions]
            texts = [x.text for x in predictions]
            scores = [x.score for x in predictions]

            img = draw_ocr(img, dt_boxes, texts, scores, font_path=str(font_path))
        return img, results


class WebsocketRec(Rec):
    async def inference(self, img: ndarray):
        pass

