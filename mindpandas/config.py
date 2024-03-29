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
"""Mindpandas config file"""
import mindpandas.backend.eager.eager_backend as eager_backend
import mindpandas.internal_config as i_config

__all__ = ['set_concurrency_mode', 'get_concurrency_mode', 'set_partition_shape', 'get_partition_shape',
           'set_benchmark_mode', 'get_benchmark_mode',
           'set_adaptive_concurrency', 'get_adaptive_concurrency', 'set_min_block_size', 'get_min_block_size',
           'set_lazy_mode', 'is_lazy_mode', 'get_process_mode']


def set_concurrency_mode(mode, **kwargs):
    """
    Set the backend concurrency mode to parallelize the computation. Default mode is ``'multithread'``.
    Available options are ``'multithread'``, ``'multiprocess'``. For the instruction and usage of
    two modes, please referring to `MindPandas execution mode introduction and configuration instructions
    <https://www.mindspore.cn/mindpandas/docs/zh-CN/master/mindpandas_configuration.html>`_
    for more information.

    Args:
        mode(str): This parameter can be set to ``'multithread'`` for multithread backend,
            or ``'multiprocess'`` for distributed multiprocess backend.
        **kwargs: When running on ``multithread`` mode, no additional parameters are required.
            When running on ``multiprocess`` mode, additional parameters include:

            * address(str): The ip address of the master node. Optional, uses ``"127.0.0.1"`` by default.
            * cpu(int): The number of CPU cores to use. Optional, uses all CPU cores by default.
            * datamem(int): The amount of memory used by datasystem (MB). Optional, uses 30% of total memory by default.
            * mem(int): The total memory (including datamem) used by MindPandas (MB).
              Optional, uses 90% of total memory by default.
            * tmp_dir(str): The temporary directory for the mindpandas process.
              Optional, uses ``"/tmp/mindpandas"`` by default.
            * tmp_file_size_limit(int): The temporary file size limit (MB).
              Optional, the default value is "None" which uses up to 95% of current free disk space.


    Raises:
        ValueError: If `mode` is not ``'multithread'`` or ``'multiprocess'``.

    Examples:
        >>> # Change the mode to multiprocess.
        >>> import mindpandas as pd
        >>> pd.set_concurrency_mode('multiprocess')
    """
    support_mode = ['multithread', 'multiprocess']
    if mode not in support_mode:
        raise ValueError(f"Mode {mode} is not supported.")

    i_config.set_concurrency_mode(mode)

    if mode == 'multiprocess':
        eager_backend.set_yr_backend(**kwargs)
    elif mode == 'multithread':
        eager_backend.set_python_backend()


def get_concurrency_mode():
    """
    Get the current concurrency mode. It would be ``'multithread'`` or ``'multiprocess'``.

    Returns:
        str, current concurrency mode.

    Examples:
        >>> # Get the current concurrency mode.
        >>> import mindpandas as pd
        >>> mode = pd.get_concurrency_mode()
    """
    mode = i_config.get_concurrency_mode()
    return mode


def set_benchmark_mode(mode):
    """
    Users can select if they want to turn on benchmarkmode for performance analysis. Default `mode` is ``False``.

    Args:
        mode(bool): This parameter can be set to ``True`` or ``False``.

    Raises:
        ValueError: If `mode` is not bool.

    Examples:
        >>> # Change the mode to True.
        >>> import mindpandas as pd
        >>> pd.set_benchmark_mode(True)
    """
    support_mode = [True, False]
    if mode not in support_mode:
        raise ValueError("Mode {} is not supported.")

    i_config.set_benchmark_mode(mode)


def get_benchmark_mode():
    """
    Get the status of the benchmark mode in the current environment.

    Returns:
        bool, Indicates whether the benchmark mode is enabled.

    Examples:
        >>> # Get the current benchmark mode.
        >>> import mindpandas as pd
        >>> mode = pd.get_benchmark_mode()
    """
    return i_config.get_benchmark_mode()


def set_partition_shape(shape):
    """
    Users can set the partition shape of the data, where shape[0] is the expected number of partitions along axis 0 (
    row-wise) and shape[1] is the expected number of partitions along axis 1 (column-wise). e.g. If the shape is
    :math:`(16, 16)`, then mindpandas will try to slice original data into 16 * 16 partitions.

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
        tuple, Number of expected partitions along each axis. It is a tuple of two positive integers. The first element
        is the row-wise number of partitions and the second element is the column-wise number of partitions.

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
    and I have a dataframe which only has 16 columns and the partition shape is :math:`(2, 2)`,
    then during the partitioning we won't further split the columns.

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


def set_adaptive_concurrency(adaptive, **kwargs):
    """
    Users can choose whether to enable the adaptive concurrency mode.

    Args:
        adaptive(bool): Indicates whether to enable the adaptive concurrency mode. This function is enabled when this
            parameter is set to ``True``. When the size of the file read from read_csv is greater than 18 MB, when
            DataFrame initialized from "pandas.DataFrame", or when the memory usage is greater than 1 GB will use
            the multiprocess mode, otherwise multithread mode will be used. If this parameter is set to ``False``,
            the adaptive concurrency mode is disabled and the concurrency mode set in the current environment is used.
        **kwargs: When `adaptive` is set to ``False``, no additional parameters are required. When `adaptive` is set to
            ``True``, `kwargs` includes:

            * address(str): The ip address of the master node. Optional, uses ``"127.0.0.1"`` by default.
            * cpu(int): The number of CPU cores to use. Optional, uses all CPU cores by default.
            * datamem(int): The amount of memory used by datasystem (MB). Optional, uses 30% of total memory by default.
            * mem(int): The total memory (including datamem) used by MindPandas (MB).
              Optional, uses 90% of total memory by default.
            * tmp_dir(str): The temporary directory for the mindpandas process.
              Optional, uses ``"/tmp/mindpandas"`` by default.
            * tmp_file_size_limit(int): The temporary file size limit (MB).
              Optional, the default value is ``None`` which uses up to 95% of current free disk space.

    Raises:
        ValueError: if `adaptive` is not bool.

    Examples:
        >>> # Set adaptive concurrency to True.
        >>> import mindpandas as pd
        >>> pd.set_adaptive_concurrency(True)
    """
    if not isinstance(adaptive, bool):
        raise ValueError(f"adaptive must be False or True, but got {adaptive}.")

    if adaptive:
        eager_backend.set_yr_backend(**kwargs)

    i_config.set_adaptive_concurrency(adaptive)


def get_adaptive_concurrency():
    """
    Get the flag for using adaptive concurrency or not.

    Returns:
        bool, whether to apply the adaptive concurrency mode.

    Examples:
        >>> # Get the adaptive concurrency flag.
        >>> import mindpandas as pd
        >>> adaptive = pd.get_adaptive_concurrency()
    """
    return i_config.get_adaptive_concurrency()


def get_adaptive_partition_shape(mode):
    """
    Get the partition shape based on `mode` for adaptive concurrency.

    Args:
        mode(str): ``'multithread'`` or ``'multiprocess'``.

    Raises:
        ValueError: if `mode` is not ``'multithread'`` or ``'multiprocess'``.

    Returns:
        tuple, the partition shape for that mode.

    Examples:
        >>> # Get the adaptive partition shape.
        >>> import mindpandas as pd
        >>> adaptive = pd.get_adaptive_partition_shape()
    """
    support_mode = ['multithread', 'multiprocess']
    if mode not in support_mode:
        raise ValueError(f"Mode {mode} is not supported.")
    return i_config.get_adaptive_partition_shape(mode)


def is_lazy_mode():
    return i_config.is_lazy_mode()


def set_lazy_mode(flag):
    i_config.set_lazy_mode(flag)

def get_process_mode():
    """
    Get string flag to indicate the lazy process mode.

    Returns:
       string, ``'batch'`` or ``'stream'``. Only ``'batch'`` is supported now.
    """
    i_config.get_process_mode()
