import sys
from threading import Thread
from multiprocessing import Queue
from typing import Callable, Any

import cv2
from numpy import ndarray


class VideoThread(Thread):
    def __init__(self, q_camera: Queue, q_result: Queue):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

        self.q_camera: Queue = q_camera
        self.q_result: Queue = q_result

        self.hooks: list = []

    def run(self):
        while True:
            ret, frame = self.cap.read()

            if frame is None:
                continue

            frame: ndarray = cv2.resize(frame, (640, 480))

            if not self.q_camera.full() and frame is not None:
                self.q_camera.put(frame)

            if not self.q_result.empty():
                frame = self.q_result.get()
            [h(frame) for h in self.hooks]
            # cv2.imshow('Test rec', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    def on_frame(self, handler: Callable[[ndarray], Any]):
        self.hooks.append(handler)
        return self.hooks.index(handler)
