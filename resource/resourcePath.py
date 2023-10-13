# encoding: utf-8
import os
from tools.log import log

class ResourcePath():
    """
    资源路径类
    """
    def __init__(self):
        self.__pwd = os.path.dirname(__file__)
        
        # 路径依赖
        self._model_zoo = "model_zoo"
        self._test_images = "test_images"
        self._font = "font"
    
    def getResPath(self):
        log.info(self.__pwd)
        return self.__pwd
    
    def getFontPath(self):
        """
        :return: 中文字库路径
        """
        temp_path = os.path.join(self.__pwd, self._font, "simsun.ttc")
        assert os.path.exists(temp_path)
        log.info(temp_path)
        return temp_path

    def getModelPath(self):
        """
        :return: 获取模型路径
        """
        temp_path = os.path.join(self.__pwd, self._model_zoo, "face_mask_model_rfb.tflite")
        assert os.path.exists(temp_path)
        log.info(temp_path)
        return temp_path

sourcePath = ResourcePath()

if __name__ == '__main__':
    res = sourcePath
    print(res.getResPath())
    print(res.getFontPath())
    print(res.getModelPath())
