import hooks
import logging
import time
from multiprocessing import Queue
from threading import Thread
from typing import Optional, Any
import colorlog

from base.components.face import FaceDetectRec, FaceRecParams
from base.components.ocr import OcrRec, SMALL_REC_PATH, LARGE_REC_PATH, PPOCR_KEYS_PATH, DET_PATH
from base.components.rec import Rec
from modules.web import FlaskThread
from modules.video import VideoThread


class RecManager:
    class RecModuleThread(Thread):
        def __init__(self, rec: Rec, q_in: Queue, q_out: Queue):
            super().__init__()
            self.rec: Rec = rec
            self.q_in: Queue = q_in
            self.q_out: Queue = q_out

            self.inf: Optional[Any] = None
            self.interval = 0.65  # 刷新间隔
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
                        min_face_count=2
                    )
                hooks.hook_manager.emit('face_inference', len(inf))
                if len(inf) >= self.face_rec_params.min_face_count:
                    if time.time() - self.face_rec_params.last_success_time > self.face_rec_params.time_threshold:
                        logger.info(f'识别到多于{self.face_rec_params.min_face_count}人脸 [警报]')
                else:
                    self.face_rec_params.last_success_time = time.time()
            if isinstance(self.rec, OcrRec):
                pass

    def __init__(self, q_camera: Queue, q_result: Queue):
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
        self.q_result: Queue = q_result

        self.q_modules: list[Queue] = [Queue(5) for _ in range(len(self.enable_recs) - 1)]
        self.q_modules = [self.q_camera] + self.q_modules + [self.q_result]
        logger.info(self.q_modules)
        self.rec_module_threads: list[Thread] = [
            RecManager.RecModuleThread(rec, self.q_modules[i], self.q_modules[i + 1])
            for i, rec in
            enumerate(self.enable_recs)
        ]

    def start(self):
        for rec_module_thread in self.rec_module_threads:
            rec_module_thread.start()


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fmt = colorlog.ColoredFormatter(
    # '%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s',
    '%(log_color)s[%(asctime)s] [%(levelname)s]- %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
        'SUCCESS': 'green'
    })
ch = colorlog.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)


class MultiProcessorManager:

    def __init__(self):
        self.q_camera: Queue = Queue(5)
        self.q_result: Queue = Queue(5)
        self.video_thread: Optional[VideoThread] = None
        self.rec_thread: Optional[Thread] = None
        self.flask_thread: Optional[Thread] = None

    def video_run(self):
        self.video_thread = VideoThread(self.q_camera, self.q_result)
        logger.info('启动视频采集进程...')
        self.video_thread.start()

    def rec_run(self):
        self.rec_thread = RecManager(self.q_camera, self.q_result)
        logger.info('启动识别管理器...')
        self.rec_thread.start()

    def flask_run(self):
        self.flask_thread = FlaskThread(self.video_thread)
        logger.info('启动web服务...')
        self.flask_thread.start()


manager: MultiProcessorManager = MultiProcessorManager()

if __name__ == '__main__':
    def main():
        global manager
        logger.info('初始化完毕！')
        manager.video_run()
        manager.rec_run()
        manager.flask_run()


    main()
