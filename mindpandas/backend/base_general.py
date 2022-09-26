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
This module defines the BaseGeneral class which is used to implement general
functions for BaseFrame Class.
"""

class BaseGeneral:
    """
    An interface used to implement general functions for BaseFrame Class
    """
    general_module_: None
    import mindpandas.backend.eager.eager_general as general_module
    general_module_ = general_module

    @classmethod
    def concat(cls, objs, axis, obj_is_series, join, ignore_index, verify_integrity, sort, keys, names,
               levels, **kwargs):
        return cls.general_module_.concat(objs, axis, obj_is_series, join, ignore_index, verify_integrity, sort, keys,
                                          names, levels, **kwargs)
