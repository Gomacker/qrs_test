import dataclasses

import cv2
import numpy as np
from numpy import ndarray

from base.components.onnx_infer import OnnxRun
from base.components.rec import Rec
from base.components.utils.face import getFaceBoxs

FACE_DET_PATH = "./resource/model_zoo/scrfd_500m_bnkps_shape160x160.onnx"
FACE_LIBRARY_PATH = "./resource/face_library/"
FACE_NAME_PATH = "./base/components/Face_Name.py"


class FaceDetectRec(Rec):
    @dataclasses.dataclass
    class Precondition:
        box: ndarray

    def __init__(
            self,
            face_det_path=FACE_DET_PATH,
            face_path=FACE_LIBRARY_PATH,
            face_name_path=FACE_NAME_PATH
    ):
        self.onnx_run = OnnxRun(model_path=face_det_path)
        # self.face_rec = FaceRec(face_path=face_path, face_name_path=face_name_path)

        self.predictions = []

    def display(self, img: ndarray, predictions: list):
        if predictions:
            bboxes = [x.box for x in predictions]

            for i, bbox in enumerate(bboxes):
                x, y, w, h, trk_id = bbox
                cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (255, 255, 0), 2)

                # cv2.putText(
                #     img,
                #     'Face',
                #     (int(x), int(y)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1, (0, 0, 255), 2, cv2.LINE_AA
                # )
        return img, []

    async def inference(self, img: ndarray):
        img = img.copy()
        input_data = self.img_preprocessing(img)
        # print(type(input_data))
        # if isinstance(input_data, ndarray):
        #     print(input_data.shape)
        net_outs = self.onnx_run.inference(input_data)
        bboxes, _ = getFaceBoxs(img, net_outs)
        self.predictions = [FaceDetectRec.Precondition(x) for x in bboxes]
        return self.predictions

    def img_preprocessing(self, img: ndarray):
        INPUT_SIZE = (160, 160)
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(INPUT_SIZE[1]) / INPUT_SIZE[0]
        if im_ratio > model_ratio:
            new_height = INPUT_SIZE[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = INPUT_SIZE[0]
            new_height = int(new_width * im_ratio)

        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((INPUT_SIZE[1], INPUT_SIZE[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        input_size = tuple(det_img.shape[0:2][::-1])
        input_data = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128, input_size,
            (127.5, 127.5, 127.5),
            swapRB=True
        )
        return input_data
