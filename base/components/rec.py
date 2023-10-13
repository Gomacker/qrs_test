import abc
from typing import Any

from numpy import ndarray


class Rec(abc.ABC):
    @abc.abstractmethod
    async def inference(self, img: ndarray):
        pass

    @abc.abstractmethod
    def display(self, img: ndarray, predictions: list):
        pass
