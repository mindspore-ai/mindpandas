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
"""Mindpandas IO"""
import pandas
from pandas._libs import lib
from pandas._typing import StorageOptions

from .compiler.query_compiler import QueryCompiler as qc


def read_csv(
        filepath_or_buffer,
        sep=lib.no_default,
        delimiter=None,
        header="infer",
        names=lib.no_default,
        index_col=None,
        usecols=None,
        squeeze=False,
        prefix=lib.no_default,
        mangle_dupe_cols=True,
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skipinitialspace=False,
        skiprows=None,
        nrows=None,
        na_values=None,
        keep_default_na=True,
        na_filter=True,
        verbose=False,
        skip_blank_lines=True,
        parse_dates=False,
        infer_datetime_format=False,
        keep_date_col=False,
        date_parser=None,
        dayfirst=False,
        cache_dates=True,
        iterator=False,
        chunksize=None,
        compression="infer",
        thousands=None,
        decimal: str = ".",
        lineterminator=None,
        quotechar='"',
        quoting=0,
        escapechar=None,
        comment=None,
        encoding=None,
        encoding_errors="strict",
        dialect=None,
        error_bad_lines=None,
        warn_bad_lines=None,
        on_bad_lines=None,
        skipfooter=0,
        doublequote=True,
        delim_whitespace=False,
        low_memory=True,
        memory_map=False,
        float_precision=None,
        storage_options: StorageOptions = None,
):
    """read data from csv file"""
    kwargs = dict(sep=sep, delimiter=delimiter, header=header, names=names,
                  index_col=index_col, usecols=usecols, squeeze=squeeze,
                  prefix=prefix, mangle_dupe_cols=mangle_dupe_cols,
                  dtype=dtype, engine=engine, converters=converters,
                  true_values=true_values, false_values=false_values,
                  skipinitialspace=skipinitialspace, skiprows=skiprows,
                  nrows=nrows, na_values=na_values,
                  keep_default_na=keep_default_na, na_filter=na_filter,
                  verbose=verbose, skip_blank_lines=skip_blank_lines,
                  parse_dates=parse_dates,
                  infer_datetime_format=infer_datetime_format,
                  keep_date_col=keep_date_col, date_parser=date_parser,
                  dayfirst=dayfirst, cache_dates=cache_dates,
                  iterator=iterator, chunksize=chunksize,
                  compression=compression, thousands=thousands,
                  decimal=decimal, lineterminator=lineterminator,
                  quotechar=quotechar, quoting=quoting, escapechar=escapechar,
                  comment=comment, encoding=encoding,
                  encoding_errors=encoding_errors, dialect=dialect,
                  error_bad_lines=error_bad_lines,
                  warn_bad_lines=warn_bad_lines, on_bad_lines=on_bad_lines,
                  skipfooter=skipfooter, doublequote=doublequote,
                  delim_whitespace=delim_whitespace, low_memory=low_memory,
                  memory_map=memory_map, float_precision=float_precision,
                  storage_options=storage_options)

    if chunksize or iterator:
        return qc.default_to_pandas_general(pandas.read_csv, filepath_or_buffer, **kwargs)
    return qc.read_csv(filepath_or_buffer, **kwargs)


def read_feather(path, columns=None, use_threads=True, storage_options=None):
    """read data from feather file"""
    return qc.read_feather(path, columns=columns, use_threads=use_threads, storage_options=storage_options)


def from_numpy(input_array):
    """read data from numpy array"""
    return qc.from_numpy(input_array)
