# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
This module defines the BaseIO class which is used to implement IO functions
for BaseFrame Class.
"""

class BaseIO:
    """
    An interface used to implement IO functions for BaseFrame Class
    """
    io_module_: None
    import mindpandas.backend.eager.eager_io as io_module
    io_module_ = io_module

    @classmethod
    def read_csv(cls, filepath, **kwargs):
        return cls.io_module_.read_csv(filepath, **kwargs)

    @classmethod
    def read_feather(cls, filepath, **kwargs):
        return cls.io_module_.read_feather(filepath, **kwargs)

    @classmethod
    def from_numpy(cls, input_array):
        return cls.io_module_.from_numpy(input_array)

    @classmethod
    def from_pandas(cls, pandas_input):
        return cls.io_module_.from_pandas(pandas_input)

    @classmethod
    def create_backend_frame(cls, data, index, columns, dtype, copy, container_type):
        return cls.io_module_.create_backend_frame(data, index, columns, dtype, copy, container_type)

    @classmethod
    def build_series_backend_frame(cls, data, index, dtype, name, copy):
        return cls.io_module_.build_series_backend_frame(data, index, dtype, name, copy)
