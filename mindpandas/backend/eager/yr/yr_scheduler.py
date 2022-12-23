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
import multiprocessing
import os

import psutil
import yr


class YrScheduler:
    """The class provides distributed execution interface."""
    op_actors = None

    @classmethod
    def init(cls, **kwargs):
        """init yr"""
        if not yr.is_initialized():
            address = kwargs.pop("address", "127.0.0.1")
            cpu = kwargs.pop("cpu", multiprocessing.cpu_count()) * 1000
            system_memory = psutil.virtual_memory().total // (1 << 20)  # Available memory in MB.
            datamem = kwargs.pop("datamem", int(system_memory * 0.3))
            mem = kwargs.pop("mem", int(system_memory * 0.9))
            tmp_dir = kwargs.pop("tmp_dir", "/tmp/mindpandas/")
            if not os.path.isabs(tmp_dir):
                raise ValueError(f'"{tmp_dir}" is not a valid path')
            spill_path = os.path.join(tmp_dir, "mp-")
            spill_limit = kwargs.pop("tmp_file_size_limit", None)

            deployment_conf = yr.DeploymentConfig(
                cpu=cpu,
                datamem=datamem,
                mem=mem,
                spill_path=spill_path,
                spill_limit=spill_limit
            )

            conf = yr.Config(function_id="sn:cn:yrk:12345678901234561234567890123456:function:0-default-func:$latest",
                             in_cluster=True,
                             recycle_time=300,
                             server_address=address,
                             ds_address=address,
                             auto=True,
                             deployment_config=deployment_conf,
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
        result = yr.get(obj_ids, timeout=3600)
        return result

    @classmethod
    def wait_computation_finished(cls, obj_ids):
        '''Wait until computation finished.'''
        yr.wait([obj_ids])

    @classmethod
    def wait(cls, obj_ids):
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        ready_id, pending_id = yr.wait(obj_ids)
        return ready_id, pending_id

    @classmethod
    def remote(cls, apply_func, *args, **kwargs):
        future_id, meta_data_id = apply_func.invoke(*args, **kwargs)
        return future_id, meta_data_id

    @classmethod
    def shutdown(cls):
        yr.finalize()
