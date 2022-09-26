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
"""
Module for providing backend functions required for eager execution of operations.
"""
# Default backend is 'multithread' and use Partition class.
from .partition import Partition as backend_partition

backend_scheduler = ''
backend_remote_functions = ''


def set_yr_backend(**kwargs):
    """Init yr backend if users set backend as yr"""
    global backend_scheduler
    global backend_partition
    global backend_remote_functions
    import mindpandas.backend.eager.yr.yr_remote_functions as backend_remote_functions
    from .yr.yr_scheduler import YrScheduler as backend_scheduler
    from .ds_partition import DSPartition as backend_partition
    import atexit
    backend_scheduler.init(**kwargs)
    atexit.register(backend_scheduler.shutdown)


def set_python_backend():
    """Init python backend for multithreaded or singlethreaded dataframes in adaptive concurrency mode."""
    global backend_partition
    from .partition import Partition as backend_partition


def get_scheduler():
    """Return corresponding backend_scheduler"""
    global backend_scheduler
    return backend_scheduler


def get_partition():
    """Return corresponding backend_partition, DSPartition, etc"""
    global backend_partition
    return backend_partition


def remote_functions():
    """Return corresponding remote_functions"""
    global backend_remote_functions
    return backend_remote_functions
