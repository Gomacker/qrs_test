import asyncio
import multiprocessing

import cv2
from numpy import ndarray

from base.components.face import FaceDetectRec
from base.components.ocr import OcrRec, SMALL_REC_PATH, LARGE_REC_PATH, PPOCR_KEYS_PATH, DET_PATH
from base.components.rec import Rec


class EmbeddedManager:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 8080


if __name__ == '__main__':
    async def main():
        enable_recs: list = [
            OcrRec(
                DET_PATH,
                SMALL_REC_PATH,
                LARGE_REC_PATH,
                PPOCR_KEYS_PATH
            ),
            FaceDetectRec()
        ]
        cap: cv2.VideoCapture = cv2.VideoCapture(0)

        frame: ndarray

        while True:
            _, frame = cap.read()
            img: ndarray = cv2.resize(frame, (640, 480))

            for rec in enable_recs:
                inf = await rec.inference(img)
                img, ocr_str = rec.display(img, inf)

            cv2.imshow('test rec', img)

            if cv2.waitKey(1) == ord('q'):
                break


    asyncio.run(main())
