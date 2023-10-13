import asyncio
import dataclasses
import logging
import multiprocessing
import time
from multiprocessing import Queue
from threading import Thread
from typing import Optional, Any

import colorlog
import cv2
import numpy
from numpy import ndarray

from base.components.face import FaceDetectRec
from base.components.ocr import OcrRec, SMALL_REC_PATH, LARGE_REC_PATH, PPOCR_KEYS_PATH, DET_PATH
from base.components.rec import Rec


class VideoThread(Thread):
    def __init__(self, q_camera: Queue, q_rec: Queue):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

        self.q_camera: Queue = q_camera
        self.q_rec: Queue = q_rec

    def run(self):
        while True:
            ret, frame = self.cap.read()
            frame: ndarray = cv2.resize(frame, (640, 480))

            if not self.q_camera.full() and frame is not None:
                self.q_camera.put(frame)

            if not self.q_rec.empty():
                frame = self.q_rec.get()

            cv2.imshow('Test rec', frame)

            if cv2.waitKey(1) == ord('q'):
                break

@dataclasses.dataclass
class FaceRecParams:
    last_success_time: float
    time_threshold: float

    tolerance_counter = 0
    tolerance_limit = 5

    min_face_count: int


class RecManager:
    class RecModuleThread(Thread):
        def __init__(self, rec: Rec, q_in: Queue, q_out: Queue):
            super().__init__()
            self.rec: Rec = rec
            self.q_in: Queue = q_in
            self.q_out: Queue = q_out

            self.inf: Optional[Any] = None
            self.interval = 0.5
            self.last_inf_time = 0

            self.face_rec_params: Optional[FaceRecParams] = None

        def run(self):
            while True:
                if not self.q_in.empty():
                    img = self.q_in.get()
                    if self.last_inf_time + self.interval <= time.time():
                        self.last_inf_time = time.time()
                        self.inf = self.rec.inference(img)
                        self.on_inference(self.inf)
                    if self.inf is not None:
                        img, predictions = self.rec.display(img, self.inf)
                        self.q_out.put(img)

        def on_inference(self, inf: Any):
            if isinstance(self.rec, FaceDetectRec):
                if not self.face_rec_params:
                    self.face_rec_params = FaceRecParams(
                        last_success_time=0,
                        time_threshold=3,
                        min_face_count=1
                    )
                if len(inf) >= self.face_rec_params.min_face_count:
                    if time.time() - self.face_rec_params.last_success_time > self.face_rec_params.time_threshold:
                        logger.info(f'识别到多于{self.face_rec_params.min_face_count}人脸 [警报]')
                else:
                    self.face_rec_params.last_success_time = time.time()
            if isinstance(self.rec, OcrRec):
                pass

    def __init__(self, q_camera: Queue, q_rec: Queue):
        logger.info('init RecManager')

        self.enable_recs: list = [
            OcrRec(
                DET_PATH,
                SMALL_REC_PATH,
                LARGE_REC_PATH,
                PPOCR_KEYS_PATH
            ),
            FaceDetectRec()
        ]
        logger.info(self.enable_recs)

        self.q_camera: Queue = q_camera
        self.q_rec: Queue = q_rec

        self.q_modules: list[Queue] = [Queue(1) for _ in range(len(self.enable_recs) - 1)]
        self.q_modules = [self.q_camera] + self.q_modules + [self.q_rec]
        logger.info(self.q_modules)
        self.rec_module_threads: list[Thread] = [
            RecManager.RecModuleThread(rec, self.q_modules[i], self.q_modules[i + 1]) for i, rec in
            enumerate(self.enable_recs)]

    def run(self):
        for rec_module_thread in self.rec_module_threads:
            rec_module_thread.start()


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fmt = colorlog.ColoredFormatter(
    '%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    })
ch = colorlog.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)


class MultiProcessorManager:

    def __init__(self):
        self.q_camera: Queue = Queue(1)
        self.q_rec: Queue = Queue(1)
        self.video_thread: Optional[Thread] = None
        self.rec_thread: Optional[Thread] = None

    def video_run(self):
        self.video_thread = VideoThread(self.q_camera, self.q_rec)
        logger.info('启动视频采集进程...')
        self.video_thread.start()

    def rec_run(self):
        self.rec_thread = RecManager(self.q_camera, self.q_rec)
        logger.info('启动识别管理器...')
        self.rec_thread.run()


if __name__ == '__main__':
    def main():
        mpm = MultiProcessorManager()

        logger.info('初始化完毕！')
        mpm.video_run()
        mpm.rec_run()


    main()
