# Copyright 2021-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
This module defines the BaseFrame class which is used to represent a frame.
"""

from abc import ABC, abstractmethod
from mindpandas.compiler.lazy.query_plan import PlanOpMixin

class BaseFrame(ABC, PlanOpMixin):
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
