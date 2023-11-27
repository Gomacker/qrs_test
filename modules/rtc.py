from threading import Thread

from flask import Flask, Response

from .video import VideoThread


class FlaskThread(Thread):
    def __init__(self, video: VideoThread):
        super().__init__()
        self.app = Flask(__name__)
        self.init_app()

    def init_app(self):
        @self.app.get('/test')
        def _():
            return Response('')

    def run(self):
        self.app.run(port=5033)
