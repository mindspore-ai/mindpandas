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
Module that provides internal configuration parameters.
"""
_execution_mode = 'eager'
_concurrency_mode = "multithread"
_partition_shape = (16, 16)
_min_block_size = 1
_benchmark_mode = False
adaptive_concurrency = False
multithread_shape = (2, 2)
multiprocess_shape = (16, 16)

num_op_actors = 16
use_shuffle_actor = True
functions = 'pandas'


def set_concurrency_mode(mode):
    """
    For internal use only. Set the backend concurrency mode to parallelize the computation.

    Args:
        mode(str): This parameter can be set to 'multithread' or 'multiprocess'.
    """
    global _concurrency_mode
    _concurrency_mode = mode


def get_concurrency_mode():
    """
    For internal use only. Get the current concurrency mode.

    Returns:
        str, current concurrency mode.

    Raises:
        ValueError:  If adaptive_concurrency is True
    """
    global adaptive_concurrency
    if adaptive_concurrency:
        raise ValueError("adaptive_concurrency is set to True, should not call get_concurrency_mode().")

    global _concurrency_mode
    return _concurrency_mode


def set_benchmark_mode(mode):
    """
    For internal use only. Users can select if they want to turn on benchmark mode for performance analysis.
    Default mode is False.

    Args:
        mode(bool): This parameter can be set to True or False.
    """
    global _benchmark_mode
    _benchmark_mode = mode


def get_benchmark_mode():
    """
    For internal use only. Get the current benchmark mode.

    Returns:
        bool, current benchmark mode.
    """
    global _benchmark_mode
    return _benchmark_mode


def set_partition_shape(shape):
    """
    For internal use only. Set the expected partition shape of the data.

    Args:
        shape(tuple): Number of expected partitions along each axis.
    """
    global _partition_shape
    _partition_shape = shape


def get_partition_shape():
    """
    For internal use only. Get the current partition shape.

    Returns:
        tuple, current partition shape.
    """
    global _partition_shape
    return _partition_shape


def set_min_block_size(min_block_size):
    """
    For internal use only. Set the min block size of each partition.

    Args:
        min_block_size(int): Minimum size of a partition's number of rows and number of columns during partitioning.
    """
    global _min_block_size
    _min_block_size = min_block_size


def get_min_block_size():
    """
    For internal use only. Get the current min block size of each partition.

    Returns:
        int, current min block size of each partition.

    """
    global _min_block_size
    return _min_block_size


def set_adaptive_concurrency(adaptive):
    """
    For internal use only. Set the flag for using adaptive concurrency or not.

    Args:
        adaptive(bool): True or False.
    """
    global adaptive_concurrency
    adaptive_concurrency = adaptive


def get_adaptive_concurrency():
    """
    For internal use only. Get the flag for using adaptive concurrency or not.

    Returns:
        bool, value of adaptive_concurrency flag.
    """
    global adaptive_concurrency
    return adaptive_concurrency


def get_adaptive_partition_shape(mode):
    """
    For internal use only. Get the partition shape based on mode for adaptive concurrency.

    Args:
        mode(str): 'multithread' or 'multiprocess'

    Returns:
        tuple, the partition shape for that mode.
    """
    if mode == 'multiprocess':
        global multiprocess_shape
        return multiprocess_shape
    global multithread_shape
    return multithread_shape
