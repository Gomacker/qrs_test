from asyncio import Queue
from pathlib import Path
from threading import Thread

import playsound
import requests
from aiortc import VideoStreamTrack
from av.frame import Frame
from flask import Flask, Response, request
from flask_socketio import SocketIO
# from av.video.frame import VideoFrame
# from av.video import VideoStream.Frame
from numpy import ndarray

import config
from .video import VideoThread


class VideoFrameTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.queue = Queue()

    async def recv(self) -> Frame:
        frame = await self.queue.get()
        return frame

    def put_frame(self, frame: Frame):
        self.queue.put_nowait(frame)
        print(self.queue.qsize())


class FlaskThread(Thread):
    def __init__(self, video: VideoThread):
        from flask_opencv_streamer.streamer import Streamer
        super().__init__()
        self.video_thread = video
        self.streamer = Streamer(3030, False)

        def video_frame_handler(frame: ndarray):
            self.streamer.update_frame(frame)

        self.video_thread.on_frame(video_frame_handler)
        self.socketio = SocketIO()
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

        @self.socketio.on('connect')
        def handle_connect():
            print(f'{self.socketio} connected')

        self.socketio.emit('text-rcv', {'text': ''})

    def run(self):
        if not self.streamer.is_streaming:
            self.streamer.start_streaming()
