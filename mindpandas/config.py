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
"""Mindpandas config file"""
import mindpandas.backend.eager.eager_backend as eager_backend
import mindpandas.iternal_config as i_config

__all__ = ['set_concurrency_mode', 'get_concurrency_mode', 'set_partition_shape', 'get_partition_shape',
           'set_adaptive_concurrency', 'get_adaptive_concurrency', 'set_min_block_size', 'get_min_block_size']


def set_concurrency_mode(mode, **kwargs):
    """
    Set the backend concurrency mode to parallelize the computation. Default mode is multithread. Available options
    are {'multithread', 'multiprocess'}. For the instruction and usage of two modes, please referring to
    `MindPandas execution mode introduction and configuration instructions
    <https://www.mindspore.cn/mindpandas/docs/zh-CN/master/mindpandas_configuration.html>`_ for more information.

    Args:
        mode(str): This parameter can be set to 'multithread' for multithread backend, or 'multiprocess' for
            distributed multiprocess backend.
        **kwargs: When running on multithread mode, no additional kwargs needed. When running on multiprocess mode,
            additional parameters include:

            * address: The ip address of the master node, required.

    Raises:
        ValueError: If mode is not 'multithread' or 'multiprocess'.

    Examples:
        >>> # Change the mode to multiprocess.
        >>> import mindpandas as pd
        >>> pd.set_concurrency_mode('multiprocess', address='127.0.0.1')
    """
    support_mode = ['multithread', 'multiprocess']
    if mode not in support_mode:
        raise ValueError(f"Mode {mode} is not supported.")

    if mode == 'multiprocess':
        i_config.set_concurrency_mode('yr')
        i_config.set_multiprocess_backend('yr')
        address = kwargs.get('address', None)
        eager_backend.set_yr_backend(server_address=address, ds_address=address)
    elif mode == 'multithread':
        i_config.set_concurrency_mode(mode)
        eager_backend.set_python_backend()


def get_concurrency_mode():
    """
    Get the current concurrency mode. It would be one of {'multithread', 'multiprocess'}.

    Returns:
        str, current concurrency mode.

    Examples:
        >>> # Get the current concurrency mode.
        >>> import mindpandas as pd
        >>> mode = pd.get_concurrency_mode()
    """
    mode = i_config.get_concurrency_mode()
    if mode == 'yr':
        mode = 'multiprocess'
    return mode


def get_multiprocess_backend():
    """
    Get the default multiprocess backend for adaptive optimization use.

    Returns:
        str, default multiprocess backend.

    Examples:
        >>> # Get the default multiprocess backend
        >>> import mindpandas as pd
        >>> backend = pd.config.get_multiprocess_backend()
    """
    return i_config.get_multiprocess_backend()


def set_partition_shape(shape):
    """
    Users can set the partition shape of the data, where shape[0] is the expected number of partitions along axis 0 (
    row-wise) and shape[1] is the expected number of partitions along axis 1 (column-wise). e.g. If the shape is (16,
    16), then mindpandas will try to slice original data into 16 * 16 partitions.

    Args:
        shape(tuple): Number of expected partitions along each axis. It should be a tuple of two positive integers.
            The first element is the row-wise number of partitions and the second element is the column-wise number of
            partitions.

    Raises:
        ValueError: If shape is not tuple type or the value of shape is not int.

    Examples:
        >>> # Set the shape of each partition to (16, 16).
        >>> import mindpandas as pd
        >>> pd.set_partition_shape((16, 16))
    """
    if not isinstance(shape, tuple):
        raise ValueError("'shape' parameter can only be set to tuple type.")
    if len(shape) != 2:
        raise ValueError("The dimension of 'shape' parameter should be 2.")
    if not isinstance(shape[0], int) or not isinstance(shape[1], int):
        raise ValueError("Value of 'shape' parameter should be int type.")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Value of 'shape' parameter should be positive.")

    i_config.set_partition_shape(shape)


def get_partition_shape():
    """
    Get the current partition shape.

    Returns:
        shape(tuple): Number of expected partitions along each axis. It is a tuple of two positive integers.
            The first element is the row-wise number of partitions and the second element is the column-wise number of
            partitions.

    Examples:
        >>> # Get the current partition shape.
        >>> import mindpandas as pd
        >>> mode = pd.get_partition_shape()
    """
    return i_config.get_partition_shape()


def set_min_block_size(min_block_size):
    """
    Users can set the min block size of each partition using this API. It means the minimum size of each axis of each
    partition. In other words, each partition's size would be larger or equal to (min_block_size, min_block_size),
    unless the original data is smaller than this size. For example, if the min_block_size is set to be 32,
    and I have a dataframe which only has 16 columns and the partition shape is (2, 2), then during the partitioning
    we won't further split the columns.

    Args:
        min_block_size(int): Minimum size of a partition's number of rows and number of columns during partitioning.

    Raises:
        ValueError: if min_block_size is not int type.

    Examples:
        >>> # Set the min block size of each partition to 8.
        >>> import mindpandas as pd
        >>> pd.set_min_block_size(8)
    """
    if not isinstance(min_block_size, int):
        raise ValueError("'min_block_size' should be int type.")

    i_config.set_min_block_size(min_block_size)


def get_min_block_size():
    """
    Get the current min block size of each partition.

    Returns:
        int, current min_block_size of each partition in config.

    Examples:
        >>> # Get the current min block size.
        >>> import mindpandas as pd
        >>> mode = pd.get_min_block_size()
    """
    return i_config.get_min_block_size()


def set_adaptive_concurrency(adaptive):
    """
    Users can set adaptive concurrency to allow read_csv to automatically select the concurrency mode based on the
    file size. Available options are "True" or "False". When set to True, file sizes read from read_csv greater
    than 18 MB and DataFrame initialized from pandas DataFrame using more than 1 GB CPU memory will use the
    multiprocess mode, otherwise they will use the multithread mode. When set to False, it will use the current
    concurrency mode.

    Args:
        adaptive(bool): True to turn on adaptive concurrency, False to turn off adaptive concurrency.

    Raises:
        ValueError: if adaptive is not True or False.

    Examples:
        >>> # Set adaptive concurrency to True.
        >>> import mindpandas as pd
        >>> pd.set_adaptive_concurrency(True)
    """
    if adaptive not in (0, 1):
        raise ValueError(f"adaptive must be False or True, but got {adaptive}.")

    i_config.set_adaptive_concurrency(adaptive)


def get_adaptive_concurrency():
    """
    Get the flag for using adaptive concurrency or not.

    Returns:
        bool, value of adaptive_concurrency flag.

    Examples:
        >>> # Get the adaptive concurrency flag.
        >>> import mindpandas as pd
        >>> adaptive = pd.get_adaptive_concurrency()
    """
    return i_config.get_adaptive_concurrency()


def get_adaptive_partition_shape(mode):
    """
    Get the partition shape for a particular concurrency mode.

    Args:
        mode(str): 'multithread' or 'multiprocess'.

    Raises:
        ValueError: if mode is not 'multithread' or 'multiprocess'.

    Returns:
        tuple, the partition shape for that mode.

    Examples:
        >>> # Get the adaptive partition shape
        >>> import mindpandas as pd
        >>> adaptive = pd.get_adaptive_partition_shape()
    """
    support_mode = ['multithread', 'multiprocess']
    if mode not in support_mode:
        raise ValueError(f"Mode {mode} is not supported.")
    if mode == 'multiprocess':
        mode = 'yr'
    return i_config.get_adaptive_partition_shape(mode)
