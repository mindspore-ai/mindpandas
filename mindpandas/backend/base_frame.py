"""
This module defines the BaseFrame class which is used to represent a frame.
"""

from abc import ABC, abstractmethod

class BaseFrame(ABC):
    """
    An abstract class used to represent a frame
    """
    @classmethod
    def create(cls):
        from mindpandas.backend.eager.eager_frame import EagerFrame
        return EagerFrame()

    @abstractmethod
    def map(self, map_func):
        pass

    @abstractmethod
    def map_reduce(self, map_func, reduce_func, axis=0):
        pass

    @abstractmethod
    def repartition(self, output_shape, mblock_size):
        pass

    @abstractmethod
    def to_pandas(self):
        pass

    @abstractmethod
    def to_numpy(self):
        pass
