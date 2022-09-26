# Copyright 2022 Huawei Technologies Co., Ltd
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
"""This module provides scheduler abstractions based on yr."""
import yr


class YrScheduler:
    """The class provides distributed execution interface."""
    op_actors = None

    @classmethod
    def init(cls, **kwargs):
        conf = yr.Config(function_id="sn:cn:yrk:12345678901234561234567890123456:function:0-default-func:$latest",
                         in_cluster=True,
                         recycle_time=300,
                         **kwargs)
        yr.init(conf)

    @classmethod
    def put(cls, value):
        obj_id = yr.put(value)
        return obj_id

    @classmethod
    def get(cls, obj_ids):
        """Retrieve data from datasystem according to given object id"""
        if isinstance(obj_ids, tuple):
            obj_ids = list(obj_ids)
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        result = yr.get(obj_ids)
        return result

    @classmethod
    def wait(cls, obj_ids):
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        # Without time-out arg, it is synchronous and will block until all data is ready, and returns [ready_ids], [].
        # To be asynchronous, add time-out, which will return ready_ids and pending_ids list.
        ready_id, pending_id = yr.wait(obj_ids, len(obj_ids), 1)  # asynch mode
        # ready_id, pending_id = yr.wait(obj_ids, len(obj_ids)) #synch mode
        return ready_id, pending_id

    @classmethod
    def remote(cls, apply_func, *args, **kwargs):
        # yr invoke return multiple outputs as single tuple object
        future_id, meta_data_id = apply_func.invoke(*args, **kwargs)
        return future_id, meta_data_id

    @classmethod
    def shutdown(cls):
        yr.finalize()
