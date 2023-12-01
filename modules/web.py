from threading import Thread

import requests
from flask import Response, request, jsonify
from flask_socketio import SocketIO
from numpy import ndarray

import config
from hooks import hook_manager
from .video import VideoThread


class FlaskThread(Thread):
    def __init__(self, video: VideoThread):
        from flask_opencv_streamer.streamer import Streamer
        super().__init__()
        self.video_thread = video
        self.streamer = Streamer(16060, False)
        self.text_inferences = list()
        self.face_inferences = 0

        def video_frame_handler(frame: ndarray):
            self.streamer.update_frame(frame)

        self.video_thread.on_frame(video_frame_handler)
        self.socketio = SocketIO(self.streamer.flask, path='/ws')
        self.init_routes()

    def init_routes(self):
        def tts_get_token() -> str:
            response = requests.post(
                'https://aip.baidubce.com/oauth/2.0/token',
                params={
                    'client_id': config.tts_client_id,
                    'client_secret': config.tts_client_secret,
                    'grant_type': 'client_credentials'
                },
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            print(response.json())
            return response.json()['access_token']

        @self.streamer.flask.post('/tts')
        @self.streamer.flask.get('/tts')
        def _():
            response = requests.post(
                'https://tsn.baidu.com/text2audio',
                data={
                    'tex': request.values['text'],
                    'tok': tts_get_token(),
                    'cuid': config.cuid,
                    'ctp': '1',
                    'lan': 'zh',

                    'spd': '5',
                    'pit': '5',
                    'vol': '5',
                    'per': '1',
                    'aue': '3'
                }
            )
            return Response(response.content, content_type='audio/mp3')

        @hook_manager.on('text_inference')
        def _(inferences: list):
            self.text_inferences = inferences

        @hook_manager.on('face_inference')
        def _(inferences: int):
            self.face_inferences = inferences

        @self.streamer.flask.post('/info')
        def info():
            return jsonify({
                "text": self.text_inferences,
                "face": self.face_inferences
            })

        @self.socketio.on('connect')
        def handle_connect():
            print(f'{self.socketio} connected')

    def run(self):
        if not self.streamer.is_streaming:
            self.streamer.start_streaming()
