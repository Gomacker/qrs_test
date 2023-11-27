from asyncio import Queue
from threading import Thread

from flask import Flask, Response
from numpy import ndarray

from .video import VideoThread

from aiortc import VideoStreamTrack


class VideoFrameTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.queue = Queue()




class FlaskThread(Thread):
    def __init__(self, video: VideoThread):
        super().__init__()
        self.app = Flask(__name__)
        self.init_app()

        def video_frame_handler(frame: ndarray):
            frame

        video.on_frame(video_frame_handler)

    def init_app(self):
        @self.app.get('/test')
        def _():
            return Response('')

    def run(self):
        self.app.run(port=5033)
