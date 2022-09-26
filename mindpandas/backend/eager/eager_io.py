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
"""io module of eager mode"""
import logging as log
import os
from io import BytesIO

import numpy as np
import pandas
import pandas._libs.lib as lib
from pandas.io.common import infer_compression

import mindpandas as mpd
import mindpandas.backend.eager.eager_backend as eager_backend
from .eager_frame import EagerFrame

# Based on optimized FF script
adaptive_filesize_threshold = 18000000  # 18 MB
adaptive_pandas_df_memusage_threshold = 1000000000  # 1 GB


def _get_ops():
    """Return the corresponding operator according to concurrency_mode"""
    if mpd.iternal_config.get_concurrency_mode() == "yr":
        from mindpandas.backend.eager.multiprocess_operators import MultiprocessOperator as ops
    elif mpd.iternal_config.get_concurrency_mode() == "multithread":
        from mindpandas.backend.eager.multithread_operators import MultithreadOperator as ops
    else:
        from mindpandas.backend.eager.partition_operators import SinglethreadOperator as ops
    ops_ = ops
    return ops_


def _validate_args(filepath, **kwargs):
    """Validate arguments for fast_read_csv"""
    if not (isinstance(filepath, str) and os.path.exists(filepath) and infer_compression(filepath, "infer") is None):
        return False

    # In fast_read_csv, 'header' can be 'infer' or None
    header = kwargs.get('header', "infer")
    if not (header == 'infer' or header is None):
        return False

    # Other parameters need to be default values
    # refer to pandas 1.3.5 documentation
    read_csv_default_args = {
        "sep": lib.no_default,
        "delimiter": None,
        "names": lib.no_default,
        "index_col": None,
        "usecols": None,
        "squeeze": False,
        "prefix": lib.no_default,
        "mangle_dupe_cols": True,
        "dtype": None,
        "engine": None,
        "converters": None,
        "true_values": None,
        "false_values": None,
        "skipinitialspace": False,
        "skiprows": None,
        "skipfooter": 0,
        "nrows": None,
        "na_values": None,
        "keep_default_na": True,
        "na_filter": True,
        "verbose": False,
        "skip_blank_lines": True,
        "parse_dates": False,
        "infer_datetime_format": False,
        "keep_date_col": False,
        "date_parser": None,
        "dayfirst": False,
        "cache_dates": True,
        "iterator": False,
        "chunksize": None,
        "compression": 'infer',
        "thousands": None,
        "decimal": '.',
        "lineterminator": None,
        "quotechar": '"',
        "quoting": 0,
        "doublequote": True,
        "escapechar": None,
        "comment": None,
        "encoding": None,
        "encoding_errors": 'strict',
        "dialect": None,
        "error_bad_lines": None,
        "warn_bad_lines": None,
        "on_bad_lines": None,
        "delim_whitespace": False,
        "low_memory": True,
        "memory_map": False,
        "float_precision": None,
        "storage_options": None
    }
    for key, val in kwargs.items():
        if key in read_csv_default_args:
            expect_val = read_csv_default_args[key]
            if expect_val is None and val is not None:
                return False
            if expect_val != val:
                return False
    return True


def read_csv(filepath, **kwargs):
    """Read a comma-separated values (csv) file into DataFrame."""
    def default_to_pandas_read_csv(filepath, **kwargs):
        df = pandas.read_csv(filepath, **kwargs)
        partition = eager_backend.get_partition().put(data=df, coord=(0, 0))
        output_frame = EagerFrame(np.array([[partition]]))  # single partition
        output_frame = output_frame.repartition(output_frame.default_partition_shape)
        return output_frame

    if isinstance(filepath, str):
        if filepath.startswith(("http", "ftp", "s3", "gs", "file")):
            return default_to_pandas_read_csv(filepath, **kwargs)
        filepath = os.path.abspath(filepath)
        file_size = os.path.getsize(filepath)
    else:
        curpos = filepath.tell()
        file_size = filepath.seek(curpos, os.SEEK_END)
        filepath.seek(curpos)
    if mpd.config.get_adaptive_concurrency():
        if file_size < adaptive_filesize_threshold:
            eager_backend.set_python_backend()

    if _validate_args(filepath, **kwargs):
        try:
            output_frame = fast_read_csv(filepath,
                                         header=kwargs.get('header', "infer"),
                                         file_size=file_size)
            output_frame.set_index(None)
            output_frame = output_frame.repartition(output_shape=output_frame.default_partition_shape)
            return output_frame
        except Exception as err:
            log.warning("Issue with parallel read csv, defaulting back to pandas read_csv. %s", err)

    output_frame = default_to_pandas_read_csv(filepath, **kwargs)
    return output_frame


def read_feather(filepath, **kwargs):
    df = pandas.read_feather(filepath, **kwargs)
    partition = eager_backend.get_partition().put(data=df, coord=(0, 0))
    output_frame = EagerFrame(np.array([[partition]]))
    output_frame = output_frame.repartition(output_frame.default_partition_shape)
    return output_frame


def from_numpy(input_array, index, columns, dtype, copy):
    """use numpy ndarray to create mindpandas.DataFrame"""
    pandas_df = pandas.DataFrame(input_array, index=index, columns=columns, dtype=dtype, copy=copy)
    return from_pandas(pandas_df)


def from_pandas(pandas_df, container_type=pandas.DataFrame):
    """use pandas.DataFrame to create mindpandas.DataFrame"""
    if mpd.config.get_adaptive_concurrency():
        pandas_df_memusage = pandas_df.memory_usage(index=True)
        if not isinstance(pandas_df_memusage, int):
            pandas_df_memusage = pandas_df_memusage.sum()
        if pandas_df_memusage < adaptive_pandas_df_memusage_threshold:
            from mindpandas.backend.eager.multithread_operators import MultithreadOperator as mt_ops
            ops = mt_ops
            partition_shape = mpd.config.get_adaptive_partition_shape('multithread')
        else:
            from mindpandas.backend.eager.multiprocess_operators import MultiprocessOperator as mp_ops
            ops = mp_ops
            partition_shape = mpd.config.get_adaptive_partition_shape(mpd.config.get_multiprocess_backend())
    else:
        ops = _get_ops()
        partition_shape = mpd.iternal_config.get_partition_shape()
    num_rows = len(pandas_df)
    num_cols = len(pandas_df.columns) if isinstance(pandas_df, pandas.DataFrame) else 1
    row_slices, col_slices = ops.get_slicing_plan(num_rows, num_cols,
                                                  partition_shape,
                                                  mpd.iternal_config.get_min_block_size())
    output_partitions = ops.partition_pandas(pandas_df, row_slices, col_slices, container_type)
    output_frame = EagerFrame(output_partitions)
    return output_frame


def create_backend_frame(data, index, columns, dtype, copy, container_type=pandas.DataFrame):
    """create backend_frame for mindpandas.DataFrame"""
    if container_type == pandas.Series:
        if columns is not None:
            raise ValueError(f"No axis named {columns} for object type Series")
        pandas_series = pandas.Series(data, index=index, dtype=dtype, copy=copy)
        return from_pandas(pandas_series, container_type=container_type)

    if isinstance(data, np.ndarray):
        return from_numpy(data, index=index, columns=columns, dtype=dtype, copy=copy)
    if isinstance(data, mpd.Series):
        if data.name is not None:
            if lib.is_list_like(data.name):
                raise TypeError(f"unhashable type: 'list'")
            columns = [data.name]
        pandas_df = pandas.DataFrame(data.to_pandas(), index=index, columns=columns, dtype=dtype, copy=copy)
        return from_pandas(pandas_df)
    pandas_df = pandas.DataFrame(data, index=index, columns=columns, dtype=dtype, copy=copy)
    return from_pandas(pandas_df)


def read_csv_mapfn(df_meta, **kwargs):
    """Distributed read_csv function mapped to each partition"""
    df_meta = df_meta.iloc[0]
    f = open(df_meta.filepath, 'rb')
    f.seek(df_meta.start, 0)
    chunk = f.read(df_meta.end - df_meta.start)
    df_first_row = kwargs.pop('df_first_row')
    header_none = isinstance(df_first_row, pandas.DataFrame)
    if header_none:
        df_first_row = df_first_row.to_string()
        list_string = df_first_row.split()
        # get the column names and add it to each partition
        df_first_row = ','.join(list_string[:len(list_string) // 2])
        df_first_row = df_first_row + '\n'
        df_first_row = df_first_row.encode()
        chunk = df_first_row + chunk
    chunk_b = BytesIO(chunk)
    f.close()

    try:
        df = pandas.read_csv(chunk_b, header=df_meta["header"])
        df_num_cols = len(df.columns)
        if len(df.columns) == len(df_meta["column_names"]):
            df.columns = df_meta["column_names"]
        else:
            df.columns = df_meta["column_names"][0:df_num_cols]
            df.reindex(df_meta["column_names"], axis="columns")
        if not df.dtypes.equals(df_meta["column_types"]):
            raise TypeError("Columns have mixed types")
        if header_none:
            # in header==None case,
            # get rid of the first row (i.e. column names) being attached to each partition previously
            return df.iloc[1:]
        return df
    except Exception as e:
        return e


def find_next_newline(offset, f):
    f.seek(offset)
    partial_line = f.readline()
    offset = offset + len(partial_line)
    partial_line = f.readline()
    offset = offset + len(partial_line)
    return offset


def fast_make_splits(file_path, file_size=None):
    """
    Make split_points. We create a split between 2 consecutive split_points.
    So if split_points is [-1, 5, 15], we create 2 splits -> [0, 5] and [6, 15]
    """
    if file_size is None:
        file_size = os.path.getsize(file_path)
    if mpd.config.get_adaptive_concurrency():
        if file_size < adaptive_filesize_threshold:
            partition_shape = mpd.config.get_adaptive_partition_shape('multithread')
        else:
            partition_shape = mpd.config.get_adaptive_partition_shape(mpd.config.get_multiprocess_backend())
    else:
        partition_shape = mpd.iternal_config.get_partition_shape()
    num_chunks = partition_shape[0]
    chunk_size = max(1, int(file_size / num_chunks))
    offset = chunk_size
    split_points = [0]
    with open(file_path, 'rb') as f:
        for _ in range(num_chunks):
            offset = find_next_newline(offset, f)
            # Check if the newline was at end of file
            if offset < file_size:
                split_points.append(offset)
                offset += chunk_size
            if offset >= file_size:
                split_points.append(file_size)
                break

    return split_points


def fast_read_csv(file_path, header, file_size=None, **kwargs):
    """
    optimized version of read_csv()
    Only works for ASCII CSV files
    For now, supports only 1 parameter - header
    """

    split_points = fast_make_splits(file_path, file_size=file_size)

    # read first 3 rows to get column types and header info
    min_number_rows = 3
    df_head = pandas.read_csv(file_path, header=header, nrows=min_number_rows)
    cols = df_head.columns
    dtypes = df_head.dtypes

    # for header==None, get the first row(including column names) first,
    # and attach it to each partition to get the correct result
    if header is None:
        df_first_row = pandas.read_csv(file_path, nrows=1)
    else:
        df_first_row = None

    df_meta = pandas.DataFrame(None, columns=["start", "end", "column_names", "column_types", "filepath", "header"])
    for i in range(len(split_points) - 1):
        local_header = None if (i > 0) else header
        df_meta.loc[i] = [split_points[i], split_points[i + 1], cols, dtypes, file_path, local_header]

    meta_partition = eager_backend.get_partition().put(data=df_meta, coord=(0, 0))
    meta_frame = EagerFrame(np.array([[meta_partition]]))  # single partition
    meta_frame = meta_frame.repartition(output_shape=(len(df_meta), 1), mblock_size=1)

    kwargs["df_first_row"] = df_first_row
    frame = meta_frame.map(read_csv_mapfn, repartition=True, **kwargs)

    for row_part in frame.partitions:
        if not row_part[0].valid:
            raise row_part[0].get().squeeze()

    return frame


def build_series_backend_frame(data, index, dtype, name, copy):
    """create backend_frame for mindpandas.Series"""
    if isinstance(data, pandas.Series):
        # TODO: If index/dtype/name is given, we need to create a new series
        if index or dtype or name:
            data = pandas.Series(data, index, dtype, name, copy)
        pandas_df = pandas.DataFrame(data)
        return from_pandas(pandas_df, container_type=pandas.Series)
    if isinstance(data, mpd.Series):
        # TODO: If index/dtype/name specified, need to do the selection
        return data.copy(deep=True)
    pandas_df = pandas.DataFrame(pandas.Series(data, index=index, dtype=dtype, name=name, copy=copy))
    return from_pandas(pandas_df, container_type=pandas.Series)
