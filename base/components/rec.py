import abc

from numpy import ndarray


class Rec(abc.ABC):
    @abc.abstractmethod
    def inference(self, img: ndarray):
        pass

    @abc.abstractmethod
    def display(self, img: ndarray, predictions: list):
        pass

