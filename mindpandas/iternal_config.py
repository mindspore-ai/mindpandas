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
adaptive_concurrency = False
multithread_shape = (2, 2)
multiprocess_shape = (16, 16)
multiprocess_backend = 'yr'

num_op_actors = 16
use_shuffle_actor = True
functions = 'pandas'


def set_multiprocess_backend(backend):
    """
    Set the default multiprocess backend for adaptive optimization use.

    Args:
        backend(str): 'yr'

    Raises:
        ValueError: If mode is not 'yr'

    Examples:
        >>> # Change the multiprocess backend to 'yr'
        >>> mindpandas.config.set_multiprocess_backend('yr')
    """
    global multiprocess_backend
    multiprocess_backend = backend


def get_multiprocess_backend():
    """
    Get the default multiprocess backend for adaptive optimization use.

    Returns:
        str, default multiprocess backend.

    Examples:
        >>> # Get the default multiprocess backend
        >>> backend = mindpandas.config.get_multiprocess_backend
    """
    global multiprocess_backend
    return multiprocess_backend


def set_concurrency_mode(mode):
    """
    Users can select which mode they want to use to parallelize the computation. Default strategy is multithread.

    Note:
        Lazy mode is a beta version and still in development.

    Args:
        mode(str): This parameter can be set to 'multithread' or 'yr'.

    Raises:
        ValueError: If mode is not 'multithread' or 'yr'.
    """
    global _concurrency_mode
    _concurrency_mode = mode


def get_concurrency_mode():
    """
    Get the current concurrency mode.

    Returns:
        str, current concurrency mde.
    """
    global adaptive_concurrency
    if adaptive_concurrency:
        raise ValueError("adaptive_concurrency is set to True, should not call get_concurrency_mode().")

    global _concurrency_mode
    return _concurrency_mode


def set_partition_shape(shape):
    """
    Users can set the partition shape of each partition. If the shape is (16, 16),
    that means each partition has 16 columns and 16 rows. If all data only has 15 columns,
    each partition has 15 columns and 16 rows.

    Args:
        shape(tuple): Shape of each partition.

    Raises:
        ValueError: If shape is not tuple type or the value of shape is not int.

    Examples:
        >>> # Set the shape of each partition to (16, 16).
        >>> mindpandas.config.set_partition_shape((16, 16))
    """
    global _partition_shape
    _partition_shape = shape


def get_partition_shape():
    """
    Get the current partition shape.

    Returns:
        tuple, current partition shape.

    Examples:
        >>> # Get the current partition shape.
        >>> mode = mindpandas.config.get_partition_shape
    """
    global _partition_shape
    return _partition_shape


def set_min_block_size(min_block_size):
    """
    Users can set the min block size of each partition. If the partition shape is (16, 16)
    and the min block size is 8, that means the shape of each partition is (8, 8).

    Args:
        min_block_size(Int): Shape of each partition.

    Raises:
        ValueError: if min_block_size is not int type.

    Examples:
        >>> # Set the min block size of each partition to 8.
        >>> mindpandas.config.set_min_block_size(8)
    """
    global _min_block_size
    _min_block_size = min_block_size


def get_min_block_size():
    """
    Get the current min block size of each partition.

    Returns:
        int, current min block size of each partition.

    Examples:
        >>> # Get the current min block size.
        >>> mode = mindpandas.config.get_min_block_size
    """
    global _min_block_size
    return _min_block_size


def set_adaptive_concurrency(adaptive):
    """
    Set the flag to use adaptive concurrency.

    Args:
        adaptive(bool): True or False.

    Raises:
        ValueError: if adaptive is not 1 or 0.

    Examples:
        >>> # Set the adaptive concurrency flag to True.
        >>> mindpandas.config.set_adaptive_concurrency(True)
    """
    global adaptive_concurrency
    adaptive_concurrency = adaptive


def get_adaptive_concurrency():
    """
    Get the flag to use adaptive concurrency.

    Returns:
        bool, value of adaptive_concurrency flag.

    Examples:
        >>> # Get the adaptive concurrency flag
        >>> adaptive = mindpandas.config.get_adaptive_concurrency
    """
    global adaptive_concurrency
    return adaptive_concurrency


def get_adaptive_partition_shape(mode):
    """
    Get the partition shape for a particular concurrency mode.

    Args:
        mode(str): 'multithread' or 'yr'

    Raises:
        ValueError: if mode is not 'multithread' or 'yr'

    Returns:
        tuple, the partition shape for that mode.

    Examples:
        >>> # Get the adaptive partition shape
        >>> adaptive = mindpandas.config.get_adaptive_partition_shape
    """
    if mode == 'yr':
        global multiprocess_shape
        return multiprocess_shape
    global multithread_shape
    return multithread_shape
