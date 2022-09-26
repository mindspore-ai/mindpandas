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
This module defines DataFrame class which is used to process two-dimensional tabular data.
"""

import inspect
import logging as log
from typing import Iterator
import warnings

import numpy as np
import pandas
from pandas._libs.lib import no_default, is_list_like, is_scalar
from pandas.core.common import apply_if_callable
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.indexing import convert_to_index_sliceable
from pandas.util._validators import validate_bool_kwarg

import mindpandas as mpd
from mindpandas.backend.base_frame import BaseFrame
from . import iternal_config as config
from .iterator import DataFrameIterator
from .util import is_full_grab_slice
from .util import hashable

_novalue = object()


class DataFrame:
    """
    This class is used to process two-dimensional tabular data.
    """
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None):
        # Materialize the input data into a DataFrame
        if isinstance(data, BaseFrame):
            self.backend_frame = data
        else:
            from .compiler.query_compiler import QueryCompiler as qc
            self.backend_frame = qc.create_backend_frame(data, index=index, columns=columns, dtype=dtype, copy=copy)

        from .compiler.query_compiler import QueryCompiler as qc
        self._qc = qc

    @property
    def index(self):
        return self.backend_frame.index_from_partitions()

    @index.setter
    def index(self, new_index):
        self.set_axis(labels=new_index, axis=0, inplace=True)

    @property
    def columns(self):
        return self.backend_frame.columns_from_partitions()

    @columns.setter
    def columns(self, new_columns):
        self.set_axis(labels=new_columns, axis=1, inplace=True)

    def _validate_set_axis(self, axis, new_labels):
        """Check if the parameter is valid"""
        if axis not in [0, 1, 'index', 'columns']:
            raise ValueError(f"No axis named {axis} for object type DataFrame.")

        if not is_list_like(new_labels):
            raise ValueError(f"Index(...) must be called with a collection of some kind, "
                             f"'%s' was passed." % new_labels)
        label_len = len(new_labels)
        if axis == 0:
            row_nums = self.backend_frame.num_rows
            if not label_len == row_nums:
                raise ValueError(f"Length mismatch: Expected axis has {row_nums} elements, "
                                 f"new values have {label_len} elements.")
        if axis == 1:
            col_nums = self.backend_frame.num_cols
            if not label_len == col_nums:
                raise ValueError(f"Length mismatch: Expected axis has {col_nums} elements, "
                                 f"new values have {label_len} elements.")

    def set_axis(self, labels, axis=0, inplace=False):
        self._validate_set_axis(axis=axis, new_labels=labels)
        return self._qc.set_axis(input_dataframe=self, labels=labels, axis=axis, inplace=inplace)

    @property
    def values(self):
        return self.to_numpy()

    @property
    def shape(self):
        return self.backend_frame.shape

    @property
    def empty(self):
        num_rows, num_cols = self.backend_frame.shape
        return num_rows == 0 or num_cols == 0

    def _validate_dtypes(self, numeric_only=False):
        """
        Check if the dtypes of each column are the same.
        Args:
        numeric_only: bool, default is False. Whether or not to allow only the numeric data.
            If numeric_only is True and non-numeric data is found, exception will be raised.
        """
        dtype = self.dtypes[0]
        for tp in self.dtypes:
            if numeric_only and not is_numeric_dtype(tp):
                raise TypeError(f"{tp} is not a numeric data type.")
            if not numeric_only and tp != dtype:
                raise TypeError(f"Cannot compare type '{tp}' with '{dtype}'")

    def _statistic_operation(self, op_name, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        """
        Do common statistic reduce operations under frame.
        Args:
            op_name : str. Name of method to apply.
            axis : int or str. Axis to apply method on.
            skipna : bool. Exclude NA/null values when computing the result.
            level : int or str. If specified `axis` is a MultiIndex, applying method along a particular level,
                collapsing into a Series.
            numeric_only : bool, optional. Include only float, int, boolean columns. If None, will attempt to use
                everything, then use only numeric data.
            **kwargs : dict. Additional keyword arguments to pass to `op_name`.
        Returns:
            scalar, Series or DataFrame
        """
        axis = self._get_axis_number(axis)

        if level is not None:
            return self._qc.default_to_pandas(df=self, df_method=op_name, axis=axis, skipna=skipna, level=level,
                                              numeric_only=numeric_only, **kwargs)

        if not numeric_only or not isinstance(numeric_only, bool):
            try:
                self._validate_dtypes(numeric_only=True)
            except TypeError:
                if numeric_only is not None:
                    raise
            else:
                numeric_only = False

        data = (
            self._get_numeric_data(axis)
            if numeric_only is None or numeric_only
            else self
        )

        output_dataframe = getattr(self._qc, op_name)(data, axis, skipna, level, numeric_only, **kwargs)
        return output_dataframe

    def mean(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        return self._statistic_operation("mean", axis, skipna, level, numeric_only, **kwargs)

    def fillna(self,
               value=None,
               method=None,
               axis=None,
               inplace=False,
               limit=None,
               downcast=None):
        """
        Fill NA/NaN values using the specified method.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)
        if isinstance(value, (list, tuple)):
            raise TypeError(
                ' "value" parameter must be a scalar or dict, but '
                'you passed a "{0}"'.format(type(value).__name__)
            )
        if isinstance(value, (DataFrame, mpd.Series)):
            value = value.to_pandas()
        if value is None and method is None:
            raise ValueError("must specify a fill method or value")
        if value is not None and method is not None:
            raise ValueError("cannot specify both a fill method or value")
        if method is not None and method not in ["backfill", "bfill", "pad", "ffill"]:
            expecting = "pad (fill) or backfill (bfill)"
            msg = "Invalid fill method. Expecting {expecting}. Got {method}".format(
                expecting=expecting, method=method
            )
            raise ValueError(msg)
        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("limit must be an integer")
            if limit <= 0:
                raise ValueError("limit must be greater than 0")
        output_dataframe = self._qc.fillna(
            input_dataframe=self,
            squeeze_self=False,
            value=value,
            method=method,
            axis=axis,
            limit=limit,
            downcast=downcast,
        )

        if inplace:
            self.backend_frame = output_dataframe.backend_frame
            return None

        return output_dataframe

    def __repr__(self):
        return self.backend_frame.to_pandas().__repr__()

    def __len__(self):
        return self.backend_frame.num_rows

    def to_pandas(self):
        return self.backend_frame.to_pandas()

    def to_parallel(self):
        return self._qc.to_parallel(self)

    def align(self,
              other,
              join='outer',
              axis=None,
              level=None,
              copy=True,
              fill_value=None,
              method=None,
              limit=None,
              fill_axis=0,
              broadcast_axis=None,
              ):
        """
        Align two objects on their axes with the specified join method.
        """
        if isinstance(other, (mpd.Series, DataFrame)):
            other_dataframe = other.to_pandas()
        else:
            other_dataframe = other

        output_dataframe_left, output_dataframe_right = self._qc.default_to_pandas(df=self,
                                                                                   df_method="align",
                                                                                   other=other_dataframe,
                                                                                   join=join,
                                                                                   axis=axis,
                                                                                   level=level,
                                                                                   copy=copy,
                                                                                   fill_value=fill_value,
                                                                                   method=method,
                                                                                   limit=limit,
                                                                                   fill_axis=fill_axis,
                                                                                   broadcast_axis=broadcast_axis)
        return (output_dataframe_left, output_dataframe_right)

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """
        Return True if all elements are True.
        """
        if level is not None:
            if bool_only is not None:
                raise NotImplementedError("Option bool_only is not implemented with option level.")
            if not self._qc.has_multiindex(self, axis) and (level > 0 or level < -1) and level != self.index.name:
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")
        if axis is None:
            # Reduce along one dimension into series
            result = self._qc.all(self, axis=0, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

            # Reduce series to single bool
            return result.to_pandas().all(bool_only=bool_only, skipna=skipna, level=level, **kwargs)

        axis = self._get_axis_number(axis)
        return self._qc.all(self, axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """
        Return True if any element is True.
        """
        if level is not None:
            if bool_only is not None:
                raise NotImplementedError("Option bool_only is not implemented with option level.")
            if not self._qc.has_multiindex(self, axis) and (level > 0 or level < -1) and level != self.index.name:
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")
        if axis is None:
            # Reduce along one dimension into series
            result = self._qc.any(self, axis=0, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

            # Reduce series to single bool
            return result.to_pandas().any(bool_only=bool_only, skipna=skipna, level=level, **kwargs)

        axis = self._get_axis_number(axis)
        return self._qc.any(self, axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwargs):
        """
        Apply a function along an axis of the DataFrame.
        """
        if axis is None:
            raise ValueError("No axis named None for object type DataFrame")
        axis = self._get_axis_number(axis)
        if result_type not in [None, "reduce", "broadcast", "expand"]:
            raise ValueError(
                "invalid value for result_type, must be one of {None, 'reduce', 'broadcast', 'expand'}"
            )

        if isinstance(func, str) and hasattr(self, func):
            df_method = getattr(self, func)
            params = inspect.signature(df_method).parameters
            if "axis" in params.keys():
                result = df_method(axis=axis, **kwargs)
            else:
                result = df_method(**kwargs)
            return result

        outer_kwargs = dict()
        outer_kwargs['func'] = func
        outer_kwargs['axis'] = axis
        outer_kwargs['raw'] = raw
        outer_kwargs['result_type'] = result_type
        outer_kwargs['args'] = args
        outer_kwargs.update(kwargs)

        result = self._qc.apply(self, **outer_kwargs)

        return result

    def _get_axis_number(self, axis):
        if axis is no_default:
            axis = None
        return pandas.DataFrame._get_axis_number(axis) if axis is not None else 0

    def _get_axis_name(self, axis):
        return pandas.DataFrame._get_axis_name(axis)

    def to_numpy(self, dtype=None, copy=False, na_value=no_default):
        if na_value is not no_default:
            output_dataframe = self.fillna(value=na_value)
        else:
            output_dataframe = self
        return output_dataframe.backend_frame.to_numpy(dtype, copy, no_default)

    def droplevel(self, level, axis=0):
        output_dataframe = self._qc.default_to_pandas(df=self, df_method=self.droplevel, level=level, axis=axis)
        return output_dataframe

    def to_csv(self,
               path_or_buf=None,
               sep=',',
               na_rep='',
               float_format=None,
               columns=None,
               header=True,
               index=True,
               index_label=None,
               mode='w',
               encoding=None,
               compression='infer',
               quoting=None,
               quotechar='"',
               line_terminator=None,
               chunksize=None,
               date_format=None,
               doublequote=True,
               escapechar=None,
               decimal='.',
               errors='strict',
               storage_options=None):
        """
        Write object to a comma-separated values (csv) file.
        """
        if chunksize is not None:
            raise NotImplementedError("mindpandas.EagerFrame.to_csv() does not implement chunksize yet")
        if storage_options is not None:
            raise NotImplementedError("mindpandas.EagerFrame.to_csv() does not implement storage_options yet")

        kwargs = dict(path_or_buf=path_or_buf, sep=sep, na_rep=na_rep,
                      float_format=float_format, columns=columns,
                      header=header, index=index, index_label=index_label,
                      mode=mode, encoding=encoding, compression=compression,
                      quoting=quoting, quotechar=quotechar,
                      line_terminator=line_terminator, chunksize=chunksize,
                      date_format=date_format, doublequote=doublequote,
                      escapechar=escapechar, decimal=decimal, errors=errors,
                      storage_options=storage_options)

        compression_type = compression['method'] if isinstance(compression, dict) else compression

        if compression_type == 'zip':
            return self._qc.default_to_pandas(df=self, df_method="to_csv", **kwargs)

        return self._qc.to_csv(df=self, **kwargs)

    def __array__(self):
        return self.to_numpy()

    def __array_wrap__(self, result, context=None):
        if context is None:
            pass
        return self._qc.from_numpy(result)

    def groupby(
            self,
            by=None,
            axis=_novalue,
            level=None,
            as_index=True,
            sort=True,
            group_keys=True,
            squeeze: bool = no_default,
            observed=False,
            dropna=True,
    ):
        """
        Group DataFrame using a mapper or by a Series of columns.
        """
        if squeeze is not no_default:
            log.warning(
                "The `squeeze` parameter is deprecated in pandas 1.1.0."
            )
        else:
            squeeze = False

        if axis is _novalue:  # axis is not passing then default to 0, if passing None to axis then None
            axis = 0
        if axis not in (0, 1, 'index', 'columns'):
            raise ValueError(f"No axis named {axis} for object type DataFrame")

        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")

        axis = self._get_axis_number(axis)

        return self._qc.groupby(
            input_dataframe=self,
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            squeeze=squeeze,
            observed=observed,
            dropna=dropna,
        )

    def sum(self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0, **kwargs):
        """
        Return the sum of the values over the requested axis.
        """
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)

        if level is not None:
            return self._qc.default_to_pandas(df=self, df_method=self.sum, axis=axis, skipna=skipna, level=level,
                                              numeric_only=numeric_only, min_count=min_count, **kwargs)

        if not isinstance(numeric_only, bool):
            numeric_only = False

        data = self._get_numeric_data(axis) if numeric_only else self

        output_dataframe = self._qc.sum(input_dataframe=data, axis=axis, skipna=skipna, numeric_only=numeric_only,
                                        min_count=min_count,
                                        **kwargs)
        return output_dataframe

    def max(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        """
        Return the maximum of the values over the requested axis.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only is True:
            return self._qc.default_to_pandas(df=self, df_method="max", axis=axis, skipna=skipna, level=level,
                                              numeric_only=numeric_only, **kwargs)
        result = self._qc.max(self, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)
        return result

    def min(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        """
        Return the minimum of the values over the requested axis.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only is True:
            return self._qc.default_to_pandas(df=self, df_method="min", axis=axis, skipna=skipna, level=level,
                                              numeric_only=numeric_only, **kwargs)
        result = self._qc.min(self, axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)
        return result

    def std(self, axis=None, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
        return self._statistic_operation("std", axis, skipna, level, numeric_only, ddof=ddof, **kwargs)

    def add(self, other, axis='columns', level=None, fill_value=None):
        return self._math_op("add", other, axis, level, fill_value)

    def sub(self, other, axis='columns', level=None, fill_value=None):
        return self._math_op("sub", other, axis, level, fill_value)

    def mul(self, other, axis='columns', level=None, fill_value=None):
        return self._math_op("mul", other, axis, level, fill_value)

    def div(self, other, axis='columns', level=None, fill_value=None):
        return self._math_op("div", other, axis, level, fill_value)

    def truediv(self, other, axis='columns', level=None, fill_value=None):
        return self._math_op("truediv", other, axis, level, fill_value)

    def floordiv(self, other, axis='columns', level=None, fill_value=None):
        return self._math_op("floordiv", other, axis, level, fill_value)

    def mod(self, other, axis='columns', level=None, fill_value=None):
        return self._math_op("mod", other, axis, level, fill_value)

    def pow(self, other, axis='columns', level=None, fill_value=None):
        return self._math_op("pow", other, axis, level, fill_value)

    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __truediv__ = truediv
    __floordiv__ = floordiv
    __mod__ = mod
    __pow__ = pow

    def _math_op(self, op, other, axis='columns', level=None, fill_value=None):
        """
        Actually do the math operation.
        """
        if op not in ["add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow"]:
            raise NotImplementedError("Operation not supported")

        if level is None:
            if is_scalar(other):
                return self._qc.math_op(self, op, other, axis, level, fill_value)

            if isinstance(other, (mpd.DataFrame, pandas.DataFrame)):
                if self.columns.equals(other.columns):
                    return self._qc.math_op(self, op, other, axis, level, fill_value)

        # Other situations use pandas for better performance
        return self._qc.default_to_pandas(df=self, df_method=op, other=other, axis=axis, level=level,
                                          fill_value=fill_value)

    def rename(self, mapper=None, *, index=None, columns=None, axis=None, copy=True, inplace=False, level=None,
               errors='ignore'):
        """
        Alter axes labels.
        """
        if (mapper is not None and (columns is not None or index is not None)):
            raise TypeError("Cannot specify both 'mapper' and any of 'index' or 'columns'")

        if (axis is not None and (columns is not None or index is not None)):
            raise TypeError("Cannot specify both 'axis' and any of 'index' or 'columns'")

        if (columns is not None or index is not None):
            if columns is not None:
                output_dataframe = self._qc.rename(input_dataframe=self, mapper=mapper, index=None, columns=columns,
                                                   axis=axis,
                                                   copy=copy, inplace=inplace, level=level, errors=errors)
                if index is not None:
                    output_dataframe = self._qc.rename(input_dataframe=output_dataframe, mapper=mapper, index=index,
                                                       columns=None, axis=axis,
                                                       copy=copy, inplace=inplace, level=level, errors=errors)
                    return output_dataframe

            if index is not None:
                output_dataframe = self._qc.rename(input_dataframe=self, mapper=mapper, index=index, columns=None,
                                                   axis=axis,
                                                   copy=copy, inplace=inplace, level=level, errors=errors)
        else:
            output_dataframe = self._qc.rename(input_dataframe=self, mapper=mapper, index=index, columns=columns,
                                               axis=axis,
                                               copy=copy, inplace=inplace, level=level, errors=errors)

        return output_dataframe

    def isna(self):
        output_dataframe = self._qc.isna(input_dataframe=self)
        return output_dataframe

    @property
    def dtypes(self):
        output_dataframe = self._qc.dtypes(input_dataframe=self)
        return output_dataframe

    def dtypes_from_partitions(self):
        return self.backend_frame.dtypes_from_partitions()

    def isin(self, values):
        if isinstance(values, (mpd.DataFrame, mpd.Series)):
            values = values.to_pandas()
        output_dataframe = self._qc.isin(input_dataframe=self, values=values)
        return output_dataframe

    def notna(self):
        output_dataframe = self._qc.default_to_pandas(df=self, df_method=self.notna)
        return output_dataframe

    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        """
        Remove missing values.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if is_list_like(axis):
            raise TypeError("supplying multiple axes to axis is no longer supported.")

        axis = self._get_axis_number(axis)
        if how is None and thresh is None:
            raise TypeError("must specify how or thresh")
        if how is not None and how not in ["any", "all"]:
            raise ValueError("invalid how option: %s" % how)
        if subset is not None:
            if axis == 1:
                indices = self.index.get_indexer_for(subset)
                check = indices == -1
                if check.any():
                    raise KeyError(list(np.compress(check, subset)))
            else:
                indices = self.columns.get_indexer_for(subset)
                check = indices == -1
                if check.any():
                    raise KeyError(list(np.compress(check, subset)))
        output_dataframe = self._qc.dropna(self, axis, how, thresh, subset, inplace)
        return output_dataframe

    def drop(self,
             labels=None,
             axis=0,
             index=None,
             columns=None,
             level=None,
             inplace=False,
             errors="raise"):
        """
        Drop specified labels from rows or columns.
        """
        if level is not None:
            return self._qc.default_to_pandas(df=self, df_method=self.drop, labels=labels, axis=axis, index=index,
                                              columns=columns,
                                              level=level, inplace=inplace, errors=errors)

        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError("Cannot specify both labels and index/columns")
            axis = self._get_axis_name(axis)
            axes = {axis: labels}
        elif index is not None or columns is not None:
            axes, _ = pandas.DataFrame._construct_axes_from_arguments(
                (index, columns), {}
            )
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', 'index', or 'columns'"
            )

        if "index" not in axes:
            axes["index"] = None
        if axes["index"] is not None:
            if not isinstance(axes["index"], list):
                axes["index"] = [axes["index"]]
            if errors == "raise":
                non_existant = [obj for obj in axes["index"] if obj not in self.index]
                if non_existant:
                    raise ValueError(
                        "labels {} not contained in axis".format(non_existant)
                    )
                axes["index"] = [obj for obj in axes["index"] if obj in self.index]
                if not axes["index"]:
                    axes["index"] = None

        if "columns" not in axes:
            axes["columns"] = None
        if axes["columns"] is not None:
            if not isinstance(axes["columns"], list):
                axes["columns"] = [axes["columns"]]
            if errors == "raise":
                non_existant = [obj for obj in axes["columns"] if obj not in self.columns]
                if non_existant:
                    raise ValueError(
                        "labels {} not contained in axis".format(non_existant)
                    )
            else:
                axes["columns"] = [obj for obj in axes["columns"] if obj in self.columns]
                if not axes["columns"]:
                    axes["columns"] = None

        masked_dataframe = self._qc.drop(self, index=axes["index"], columns=axes["columns"])
        return masked_dataframe

    def duplicated(self, subset=None, keep="first"):
        output_dataframe = self._qc.duplicated(self, subset=subset, keep=keep)
        return output_dataframe

    def drop_duplicates(
            self, subset=None, keep="first", inplace=False, ignore_index=False
    ):
        """
        Return DataFrame with duplicate rows removed.
        """
        if subset is not None:
            if is_list_like(subset):
                if not isinstance(subset, list):
                    subset = list(subset)
            else:
                subset = [subset]
            duplicates = self.duplicated(keep=keep, subset=subset)
        else:
            duplicates = self.duplicated(keep=keep)

        if not isinstance(ignore_index, bool):
            raise ValueError(
                f'For argument "ignore_index" expected type bool, received type {type(ignore_index).__name__}.')

        if not isinstance(inplace, bool):
            raise ValueError(f'For argument "inplace" expected type bool, received type {type(inplace).__name__}.')

        indices = duplicates.values.nonzero()[0]
        masked_dataframe = self._qc.drop(self, index=self.index[indices], ignore_index=ignore_index)

        if inplace:
            self.set_backend_frame(masked_dataframe.backend_frame)
            return None

        return masked_dataframe

    def _comp_op(self, func, other, axis, level=None):
        """
        Do the operations like equal, le, lt, etc.
        """
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)
        if func in ('eq', 'le', 'lt', 'ge', 'gt', 'ne', 'equals'):
            if isinstance(other, np.ndarray):
                if len(other.shape) > 1:
                    other = mpd.DataFrame(other)
                else:
                    return self._qc.default_to_pandas(df=self,
                                                      df_method=func,
                                                      other=other,
                                                      axis=axis,
                                                      level=level)
                return self._qc.df_comp_op(self, func, other, False, axis, level)

            if isinstance(other, (dict, list, tuple, mpd.Series, pandas.Series)):
                return self._qc.default_to_pandas(df=self,
                                                  df_method=func,
                                                  other=other,
                                                  axis=axis,
                                                  level=level)
            if isinstance(other, pandas.DataFrame):
                # if the DataFrame is Hierarchical
                if isinstance(other.index, pandas.core.indexes.multi.MultiIndex):
                    return self._qc.default_to_pandas(df=self,
                                                      df_method=func,
                                                      other=other,
                                                      axis=axis,
                                                      level=level)
                other_df = DataFrame(other)
                return self._qc.df_comp_op(self, func, other_df, False, axis, level)
            if isinstance(other, DataFrame):
                # if the DataFrame is Hierarchical
                if isinstance(other.index, pandas.core.indexes.multi.MultiIndex):
                    return self._qc.default_to_pandas(df=self,
                                                      df_method=func,
                                                      other=other,
                                                      axis=axis,
                                                      level=level)
                if not self.columns.equals(other.columns):
                    return self._qc.default_to_pandas(df=self,
                                                      df_method=func,
                                                      other=other,
                                                      axis=axis,
                                                      level=level)
                return self._qc.df_comp_op(self, func, other, False, axis, level)
            if is_scalar(other):
                return self._qc.df_comp_op(self, func, other, True, axis, level)

            raise TypeError(f"argument other of type {type(other)} is not supported.")

        raise ValueError(f"argument func {func} is not supported.")

    def eq(self, other, axis='columns', level=None):
        result = self._comp_op('eq', other, axis, level)
        return result

    def __eq__(self, other):
        if isinstance(other, (pandas.DataFrame, mpd.DataFrame)):
            if self.shape != other.shape or not self.index.equals(other.index):
                raise ValueError("ValueError: Can only compare identically-labeled DataFrame objects")
        return self._comp_op('eq', other, "columns", None)

    def le(self, other, axis='columns', level=None):
        return self._comp_op('le', other, axis, level)

    def __le__(self, other):
        if isinstance(other, (pandas.DataFrame, mpd.DataFrame)):
            if self.shape != other.shape or not self.index.equals(other.index):
                raise ValueError("ValueError: Can only compare identically-labeled DataFrame objects")
        return self._comp_op('le', other, "columns", None)

    def lt(self, other, axis='columns', level=None):
        return self._comp_op('lt', other, axis, level)

    def __lt__(self, other):
        if isinstance(other, (pandas.DataFrame, mpd.DataFrame)):
            if self.shape != other.shape or not self.index.equals(other.index):
                raise ValueError("ValueError: Can only compare identically-labeled DataFrame objects")
        return self._comp_op('lt', other, "columns", None)

    def ge(self, other, axis='columns', level=None):
        return self._comp_op('ge', other, axis, level)

    def __ge__(self, other):
        if isinstance(other, (pandas.DataFrame, mpd.DataFrame)):
            if self.shape != other.shape or not self.index.equals(other.index):
                raise ValueError("ValueError: Can only compare identically-labeled DataFrame objects")
        return self._comp_op('ge', other, "columns", None)

    def gt(self, other, axis='columns', level=None):
        return self._comp_op('gt', other, axis, level)

    def __gt__(self, other):
        if isinstance(other, (pandas.DataFrame, mpd.DataFrame)):
            if self.shape != other.shape or not self.index.equals(other.index):
                raise ValueError("ValueError: Can only compare identically-labeled DataFrame objects")
        return self._comp_op('gt', other, "columns", None)

    def ne(self, other, axis='columns', level=None):
        return self._comp_op('ne', other, axis, level)

    def __ne__(self, other):
        if isinstance(other, (pandas.DataFrame, mpd.DataFrame)):
            if self.shape != other.shape or not self.index.equals(other.index):
                raise ValueError("ValueError: Can only compare identically-labeled DataFrame objects")
        return self._comp_op('ne', other, "columns", None)

    def equals(self, other):
        """
        Test whether two objects are equal.
        """
        if isinstance(other, (pandas.DataFrame, mpd.DataFrame)):
            # check Dataframe shapes, colmuns and dtypes in advanced
            if self.shape != other.shape or not self.columns.equals(other.columns) or not self.dtypes.equals(
                    other.dtypes):
                return False
            equal = self._comp_op('equals', other, 0, None).all(axis=None)
        else:
            return self._qc.default_to_pandas(df=self,
                                              df_method='equals',
                                              other=other,
                                              )

        # Convert numpy.bool to python bool
        return bool(equal)

    def combine(self, other, func, fill_value=None, overwrite=True):
        """
        Align the two dataframes' columns first

        Then combine columnwise using func to merge columns
        """
        if not isinstance(other, DataFrame):
            raise TypeError(f"Can only combine two DataFrames, a {type(other)} was passed")

        orig_df_cols = self.columns
        orig_other_cols = other.columns

        df1, df2 = self.align(other=other)
        df_shapes = (df1.shape, df2.shape)

        output_dataframe = self._qc.combine(df1,
                                            df2,
                                            orig_df_cols,
                                            orig_other_cols,
                                            df_shapes,
                                            func,
                                            fill_value=fill_value,
                                            overwrite=overwrite)
        return output_dataframe

    def explode(self, column, ignore_index=False):
        """
        This func is only used for demo purposes.

        Expand list-like elements of a column to individual row elements, replicating index values.
        """
        df = self.reset_index(drop=True)
        result = self._qc.explode(df, column=column, ignore_index=ignore_index)
        if ignore_index:
            result = result.reset_index()
        result = result.reindex(columns=self.columns)
        return result

    def stack(self, level=-1, dropna=True):
        output_dataframe = self._qc.default_to_pandas(df=self,
                                                      df_method=self.stack,
                                                      level=level,
                                                      dropna=dropna)
        return output_dataframe

    def unstack(self, level=-1, fill_value=None):
        output_dataframe = self._qc.default_to_pandas(df=self,
                                                      df_method=self.unstack,
                                                      level=level,
                                                      fill_value=fill_value)
        return output_dataframe

    def cumsum(self, axis=None, skipna=True, **kwargs):
        axis = self._get_axis_number(axis)
        output_dataframe = self._qc.cumsum(self, axis=axis, skipna=skipna, **kwargs)
        return output_dataframe

    def cummin(self, axis=None, skipna=True, **kwargs):
        axis = self._get_axis_number(axis)
        output_dataframe = self._qc.cummin(self, axis=axis, skipna=skipna, **kwargs)
        return output_dataframe

    def cummax(self, axis=None, skipna=True, **kwargs):
        axis = self._get_axis_number(axis)
        output_dataframe = self._qc.cummax(self, axis=axis, skipna=skipna, **kwargs)
        return output_dataframe

    def cumprod(self, axis=None, skipna=True, **kwargs):
        axis = self._get_axis_number(axis)
        output_dataframe = self._qc.cumprod(self, axis=axis, skipna=skipna, **kwargs)
        return output_dataframe

    def append(
            self,
            other,
            ignore_index=False,
            verify_integrity=False,
            sort=False,
    ):
        """
        Append rows of other to the end of caller, returning a new object.
        """
        if not isinstance(other, DataFrame):
            raise TypeError(f"Can only merge DataFrame objects, a {type(other)} was passed")

        if not isinstance(ignore_index, bool):
            raise TypeError(f"ignore_index has to be a boolean value, got {type(ignore_index)}")

        if ignore_index is None:
            ignore_index = False

        if verify_integrity is None:
            verify_integrity = False

        if not isinstance(sort, bool):
            raise TypeError(f"sort has to be a boolean value, got {type(sort)}")

        if sort is None:
            sort = False

        return self._qc.append(
            input_dataframe=self,
            other=other,
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            sort=sort,
        )

    def merge(
            self,
            right,
            how='inner',
            on=None,
            left_on=None,
            right_on=None,
            left_index=False,
            right_index=False,
            sort=False,
            suffixes=('_x', '_y'),
            copy=True,
            indicator=False,
            validate=None
    ):
        """
        Merge DataFrame or named Series objects with a database-style join.
        """
        if not isinstance(right, DataFrame):
            raise TypeError(f"Can only merge Series or DataFrame objects, a {type(right)} was passed")

        if how not in ['inner', 'left', 'right', 'outer', 'cross']:
            raise TypeError("Unknown type of merge test, should be in {'inner', 'left', 'right', 'outer', 'cross'}")

        # Cases not supported yet
        if left_index or right_index or how in ['outer', 'cross']:
            return self._qc.default_to_pandas(
                self,
                self.merge,
                right=right,
                how=how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
                sort=sort,
                suffixes=suffixes,
                copy=copy,
                indicator=indicator,
                validate=validate
            )

        if on is None and left_on is None and right_on is None:
            left_cols = self.columns
            right_cols = right.columns
            common_cols = left_cols.intersection(right_cols)
            if len(common_cols) == 0:
                raise pandas.errors.MergeError(
                    "No common columns to perform merge on. "
                    f"Merge options: left_on={left_on}, "
                    f"right_on={right_on}, "
                    f"left_index={left_index}, "
                    f"right_index={right_index}"
                )
            if (
                    not left_cols.join(common_cols, how="inner").is_unique
                    or not right_cols.join(common_cols, how="inner").is_unique
            ):
                raise pandas.errors.MergeError(f"Data columns not unique: {repr(common_cols)}")
            left_on = right_on = common_cols
        elif on is not None:
            if left_on is not None or right_on is not None:
                raise pandas.errors.MergeError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )
            if left_index or right_index:
                raise pandas.errors.MergeError(
                    'Can only pass argument "on" OR "left_index" '
                    'and "right_index", not a combination of both.'
                )
        elif left_on is not None:
            if left_index:
                raise pandas.errors.MergeError(
                    'Can only pass argument "left_on" OR "left_index" not both.'
                )
            if not right_index and right_on is None:
                raise pandas.errors.MergeError('Must pass "right_on" OR "right_index".')
        elif right_on is not None:
            if right_index:
                raise pandas.errors.MergeError(
                    'Can only pass argument "right_on" OR "right_index" not both.'
                )
            if not left_index and left_on is None:
                raise pandas.errors.MergeError('Must pass "left_on" OR "left_index".')

        if on:
            if not is_list_like(on):
                on = [on]
        else:
            if left_on is not None:
                if isinstance(left_on, pandas.Index):
                    left_on = left_on.to_list()
                elif not is_list_like(left_on):
                    left_on = [left_on]
            if right_on is not None:
                if isinstance(right_on, pandas.Index):
                    right_on = right_on.to_list()
                elif not is_list_like(right_on):
                    right_on = [right_on]

        return self._qc.merge(
            left=self,
            right=right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate
        )

    def insert(self, loc, column, value, allow_duplicates=False):
        """
        Insert column into DataFrame at specified location.
        """
        if not isinstance(loc, int):
            raise TypeError(f"loc has to be an integer, got {type(loc)}")
        if loc < 0 or loc > self.backend_frame.num_cols:
            raise ValueError(f"index out of range")
        if not allow_duplicates and column in self.columns:
            raise ValueError(f"Cannot insert {column}, already exists")
        if isinstance(value, (pandas.DataFrame, mpd.DataFrame)):
            if len(value.columns) != 1:
                raise ValueError(
                    f"Wrong number of items passed {len(value.columns)}, placement implies 1"
                )
            value = value.squeeze(axis=1)

        if is_list_like(value):
            if len(value) != self.backend_frame.num_rows:
                raise ValueError(
                    f"Length of values {len(value)} does not match length of index {self.backend_frame.num_rows}")
            if isinstance(value, mpd.Series):
                value = value.to_pandas()
            elif not isinstance(value, (set, pandas.Series)):
                value = list(value)
        else:
            value = [value] * self.backend_frame.num_rows

        backend_frame = self._qc.insert(
            df=self,
            loc=loc,
            column=column,
            value=value,
            allow_duplicates=allow_duplicates,
        )

        self.backend_frame = backend_frame

    def rank(self, axis=0, method="average", numeric_only=None, na_option="keep", ascending=True, pct=False):
        """
        Compute numerical data ranks along axis.
        """
        axis = self._get_axis_number(axis)
        if numeric_only and not isinstance(numeric_only, bool):
            raise TypeError(f"numeric_only has to be a boolean value, Got {type(numeric_only)}")
        if not isinstance(ascending, bool):
            raise TypeError(f"ascending has to be a boolean value, Got {type(ascending)}")
        if not isinstance(pct, bool):
            raise TypeError(f"pct has to be a boolean value, Got {type(pct)}")
        output_dataframe = self._qc.rank(self, axis=axis, method=method, numeric_only=numeric_only, na_option=na_option,
                                         ascending=ascending, pct=pct)
        return output_dataframe

    @property
    def loc(self):
        from .index import _Loc
        return _Loc(self)

    @property
    def iloc(self):
        from .index import _ILoc
        return _ILoc(self)

    def head(self, n=5):
        return self.iloc[:n]

    def tail(self, n=5):
        if n == 0:
            return self.iloc[self.backend_frame.num_rows:]
        return self.iloc[-n:]

    def reindex(
            self,
            labels=None,
            index=None,
            columns=None,
            axis=None,
            method=None,
            copy=True,
            level=None,
            fill_value=np.nan,
            limit=None,
            tolerance=None,
    ):
        """
        Conform Series/DataFrame to new index with optional filling logic.
        """
        if copy is None:
            copy = True
            log.warning("The 'copy' argument is set to True by default in mindpandas.Dataframe.reindex.")
        if copy is not True:
            raise NotImplementedError("Copy!=True is not implemented in mindpandas.Dataframe.reindex.")

        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)

        if axis == 0 and labels is not None:
            index = labels
        elif labels is not None:
            columns = labels

        if (
                level is not None
                or (index is not None and isinstance(index, pandas.MultiIndex))
                or (columns is not None and isinstance(columns, pandas.MultiIndex))
        ):
            raise TypeError(f"MultiIndex not supported")
        output_dataframe = None
        if isinstance(tolerance, (mpd.DataFrame, mpd.Series)):
            tolerance = tolerance.to_pandas()
        if index is not None:
            if not isinstance(index, pandas.Index):
                index = pandas.Index(index)
            if not index.equals(self.index):
                output_dataframe = self._qc.reindex(self, axis=0, labels=index,
                                                    method=method, level=level,
                                                    fill_value=fill_value, limit=limit,
                                                    tolerance=tolerance)
        if output_dataframe is None:
            output_dataframe = self
        final_output_datafrane = None
        if columns is not None:
            if not isinstance(columns, pandas.Index):
                columns = pandas.Index(columns)
            if not columns.equals(self.columns):
                final_output_datafrane = self._qc.reindex(output_dataframe,
                                                          axis=1, labels=columns,
                                                          method=method, level=level,
                                                          fill_value=fill_value, limit=limit,
                                                          tolerance=tolerance)
        if final_output_datafrane is None:
            final_output_datafrane = output_dataframe
        return final_output_datafrane

    def sort_values(self,
                    by,
                    axis=0,
                    ascending=True,
                    inplace=False,
                    kind='quicksort',
                    na_position='last',
                    ignore_index=False,
                    key=None):
        """
        Sort by the values along either axis.
        """
        axis = self._get_axis_number(axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        if self._qc.has_multiindex(self, axis=axis):
            return self._qc.default_to_pandas(self,
                                              df_method=self.sort_values,
                                              by=by,
                                              axis=axis,
                                              ascending=ascending,
                                              inplace=inplace,
                                              kind=kind,
                                              na_position=na_position,
                                              ignore_index=ignore_index,
                                              key=key)

        return self._qc.sort_values(self,
                                    by=by,
                                    axis=axis,
                                    ascending=ascending,
                                    inplace=inplace,
                                    kind=kind,
                                    na_position=na_position,
                                    ignore_index=ignore_index,
                                    key=key)

    def sort_index(self,
                   axis=0,
                   level=None,
                   ascending=True,
                   inplace=False,
                   kind='quicksort',
                   na_position='last',
                   sort_remaining=True,
                   ignore_index=False,
                   key=None,
                   ):
        """
        Sort object by labels along an axis.
        """
        sorted_dataframe = self._qc.default_to_pandas(df=self,
                                                      df_method=self.sort_index,
                                                      axis=axis,
                                                      level=level,
                                                      ascending=ascending,
                                                      inplace=inplace,
                                                      kind=kind,
                                                      na_position=na_position,
                                                      sort_remaining=sort_remaining,
                                                      ignore_index=ignore_index,
                                                      key=key)
        return sorted_dataframe

    def set_index(
            self,
            keys,
            drop=True,
            append=False,
            inplace=False,
            verify_integrity=False
    ):
        """
        Set the DataFrame index using existing columns.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if not isinstance(keys, list):
            keys = [keys]

        err_msg = (
            'The parameter "keys" may be a column key, one-dimensional '
            "array, or a list containing only valid column keys and "
            "one-dimensional arrays."
        )

        missing = []
        for col in keys:
            if isinstance(col, (pandas.Index, mpd.Series, np.ndarray, list, Iterator)):
                if getattr(col, "ndim", 1) != 1:
                    raise ValueError(err_msg)
            else:
                try:
                    found = col in self.columns
                except TypeError as err:
                    raise TypeError(
                        f"{err_msg}. Received column of type {type(col)}"
                    ) from err
                else:
                    if not found:
                        missing.append(col)

        if missing:
            raise KeyError(f"None of {missing} are in the columns")

        output_dataframe = self._qc.set_index(
            self, keys, drop=drop, append=append
        )

        index = output_dataframe.index
        if verify_integrity and not index.is_unique:
            duplicates = index[index.duplicated()].unique()
            raise ValueError(f"Index has duplicate keys: {duplicates}")

        if inplace:
            self.backend_frame = output_dataframe.backend_frame
            return None

        return output_dataframe

    def squeeze(self, axis=None):
        axis = self._get_axis_number(axis) if axis is not None else None
        output = self._qc.squeeze(self, axis=axis)
        return output

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        inplace = validate_bool_kwarg(inplace, "inplace")
        return self._qc.reset_index(self, level, drop, inplace, col_level, col_fill)

    def applymap(self, func, na_action=None, **kwargs):
        """Apply a function to a Dataframe element-wise.

        Args:
            func (callable): a function that accepts and returns a scalar.
            na_action {None, 'ignore'}: If ‘ignore’, propagate NaN values, without passing them to func.
            kwargs: Additional keyword arguments to pass as keywords arguments to func.

        Returns:
            A DataFrame.

        Raises:
            ValueError: When input value for na_action is invalid.
            TypeError: When input value for func is not callable.
        """
        if na_action not in [None, 'ignore']:
            raise ValueError(f"na_action must be 'ignore' or None, Got {na_action}")
        if not callable(func):
            raise TypeError(f"the first argument must be callable")
        return self._qc.applymap(self, func, na_action, **kwargs)

    def __getattr__(self, item: str):
        return self._qc.getitem_column(self, item)

    def __getitem__(self, key):
        indexer = None
        if isinstance(key, slice) or (isinstance(key, str) and key not in self.columns):
            indexer = convert_to_index_sliceable(pandas.DataFrame(index=self.index), key)
        if indexer is not None:
            if is_full_grab_slice(indexer, sequence_len=len(self)):
                return self.copy()
            return self.iloc[indexer]
        try:
            if key in self.columns:
                return self._qc.getitem_column(self, key)
        except (KeyError, ValueError, TypeError):
            pass
        if isinstance(key, (list, np.ndarray, pandas.Index, pandas.Series, mpd.Series)):
            return self._qc.getitem_array(self, key)

        return self._qc.default_to_pandas(self, pandas.DataFrame.__getitem__, key)

    def agg(self, func=None, axis=0, *args, **kwargs):
        axis = self._get_axis_number(axis)
        result = None

        if axis == 0:
            result = self._aggregate(func, _axis=axis, *args, **kwargs)
        if result is None:
            kwargs.pop("is_transform", None)
            return self.apply(func, axis=axis, args=args, **kwargs)
        return result

    def _aggregate(self, func, *args, **kwargs):
        """
        Aggregate using one or more operations over the specified axis.
        """
        axis = kwargs.pop("_axis", 0)
        kwargs.pop("_level", None)

        if isinstance(func, str):
            kwargs.pop("is_transform", None)
            kwargs['agg_axis'] = axis
            return self._string_function(func, *args, **kwargs)

        if func is None or isinstance(func, dict):
            return self._qc.default_to_pandas(self, self.agg, func, *args, **kwargs)

        kwargs.pop("is_transform", None)
        kwargs["apply_from_agg"] = 1
        return self.apply(func, axis=axis, args=args, **kwargs)

    def _string_function(self, func, *args, **kwargs):
        f = self._qc._validate_string_func(func)
        if f is not None:
            return self._qc.default_to_pandas(self, "agg", func, *args, **kwargs)
        raise ValueError("{} is an unknown string function".format(func))

    def transpose(self, copy=None):
        """Transpose index and columns.

        Args:
            copy (bool): return a copy if True.

        Returns:
            Transposed dataframe.

        Raises:
            NotImplementedError: when value of copy is not True.
        """
        if copy is None:
            copy = True
            log.warning("The 'copy' argument is set to True by default in mindpandas.Dataframe.transpose. "
                        "Copy!=True is not implemented in mindpandas.Dataframe.transpose"
                        )
        if copy is not True:
            raise NotImplementedError("Copy!=True is not implemented in mindpandas.Dataframe")
        return self._qc.transpose(self, copy=copy)

    T = property(transpose)

    def astype(self, dtype, copy=True, errors='raise'):
        """Cast a pandas object to a specified dtype.

        Args:
            dtype (data type, dict): should be a numpy dtype or a python type.
            copy (bool): return a copy if True.
            errors {'raise', 'ignore'}: control whether to raise or ignore errors.

        Returns:
            Casted: same type as self.

        Raises:
            TypeError: when type of copy is not bool.
            ValueError: when errors got unexpected values.
            KeyError: when column names in dtype not found in self.
        """
        if not isinstance(copy, bool):
            raise TypeError(f"copy should be either True or False, got {copy}")
        if errors not in ['raise', 'ignore']:
            raise ValueError(f"value of errors should be in ['ignore', 'raise'], got {errors}")
        if isinstance(dtype, dict):
            for col_name in dtype.keys():
                if col_name not in self.columns:
                    raise KeyError(
                        "Only a column name can be used for the "
                        "key in a dtype mappings argument. "
                        f"'{col_name}' not found in columns."
                    )

        return self._qc.astype(self, dtype=dtype, copy=copy, errors=errors)

    def _set_item(self, axis, key, value):
        # update the dataframe inplace
        self._qc.setitem(self, axis, key, value)

    def _set_item_key_in_columns(self, key, value):
        """
        Set item if key is in columns.
        """
        # for simple case of mpd.Series, call quick version before checking complex cases
        if isinstance(value, mpd.Series):
            self._set_item(0, key, value)
        if is_list_like(value):
            if isinstance(value, (pandas.DataFrame, mpd.DataFrame)):
                value = value[value.columns[0]].values
            elif isinstance(value, pandas.Series):
                value = value.values
            elif isinstance(value, np.ndarray):
                if len(value.shape) >= 3:
                    warnings.warn("Shape of new values must be compartible with manager shape")

                value = value.T.reshape(-1)
                if self.backend_frame.num_rows > 0:
                    value = value[: self.backend_frame.num_rows]
        self._set_item(0, key, value)

    def _set_item_key_not_in_columns(self, key, value):
        """
        Set item if key is not in columns.
        """
        if isinstance(value, mpd.Series) and len(self.columns) == 0:
            self.backend_frame = value.backend_frame.copy(True)
            self.columns = self.columns[:-1].append(pandas.Index([key]))
            return
        if isinstance(value, (pandas.DataFrame, mpd.DataFrame)) and value.shape[1] != 1:
            raise ValueError(f"Wrong number of items passed {value.shape[1]}, placement implies 1")
        if isinstance(value, np.ndarray) and len(value.shape) > 1:
            if value.shape[1] != 1:
                raise ValueError(f"Wrong number of items passed {value.shape[1]}, placement implies 1")
            value = value.copy().T[0]

        num_rows = self.backend_frame.num_rows
        if is_scalar(value):
            value = [value] * num_rows
        value_len = len(value)
        if num_rows != value_len:
            raise ValueError(f"Length of values ({value_len}) does not match length of index ({num_rows})")
        self._qc.append_column(self, key, value)
        return

    def __setitem__(self, key, value):
        key = apply_if_callable(key, self)
        if isinstance(key, slice):
            indexer = convert_to_index_sliceable(pandas.DataFrame(index=self.index), key)
            self.iloc[indexer] = value
        if hashable(key) and key in self.columns:
            self._set_item_key_in_columns(key, value)
            return

        if hashable(key) and key not in self.columns:
            self._set_item_key_not_in_columns(key, value)
            return

        ### would use self.mask when key is mpd.DataFrame
        self._qc.default_to_pandas(self, pandas.DataFrame.__setitem__, key=key, value=value)
        return

    def __iter__(self):
        return self.to_pandas().__iter__()

    def copy(self, deep=True):
        return self._qc.copy(self, deep=deep)

    def where(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=None):
        """
        Replace values where the condition is False.
        """
        axis = self._get_axis_number(axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        try_cast = validate_bool_kwarg(try_cast, "try_cast")
        if level is not None:
            return self._qc.default_to_pandas(self,
                                              self.where,
                                              cond=cond,
                                              other=other,
                                              inplace=inplace,
                                              axis=axis,
                                              level=level,
                                              errors=errors,
                                              try_cast=try_cast)
        if not isinstance(errors, str):
            raise TypeError(f'Argument "errors" expected type str, got {type(errors)}')
        if errors not in ['raise', 'ignore']:
            raise ValueError(f'Argument "errors" should be in ["raise", "ignore"], got {errors}')
        if callable(cond):
            cond = cond(self)
        if callable(other):
            other = other(self)
        if not isinstance(cond, DataFrame):
            if not hasattr(cond, "shape"):
                cond = np.asanyarray(cond)
        if isinstance(other, DataFrame):
            if other.shape != self.shape:
                raise ValueError("Array conditional must be same shape as self")
        if cond.shape != self.shape:
            raise ValueError("Array conditional must be same shape as self")
        if not is_scalar(other) and not isinstance(other, DataFrame):
            raise TypeError(f'Argument "other" should be a scalar, DataFrame, '
                            f'or callable that returns a scalar or DataFrame')

        return self._qc.where(self, cond, other, inplace, axis, level, errors, try_cast)

    def mask(self, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=None):
        if callable(cond):
            cond_inverse = lambda *args, **kwargs: ~cond(*args, **kwargs)
        elif hasattr(cond, "__invert__"):
            cond_inverse = ~cond
        else:
            cond_inverse = ~np.array(cond)

        return self.where(
            cond_inverse,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast
        )

    def set_backend_frame(self, frame):
        if not isinstance(frame, BaseFrame):
            raise TypeError(f"Can only inplace update a DataFrame with a BaseFrame instance")
        self.backend_frame = frame

    def median(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        return self._statistic_operation("median", axis, skipna, level, numeric_only, **kwargs)

    def _get_numeric_data(self, axis: int):
        if axis != 0:
            return self
        return self.drop(
            columns=[
                i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
            ]
        )

    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
        inplace = validate_bool_kwarg(inplace, "inplace")
        if isinstance(to_replace, mpd.Series):
            to_replace = to_replace.to_pandas()
        if limit is not None and not isinstance(limit, int):
            raise TypeError(f"Parameter 'limit' should be None or an integer, got {type(limit)}")
        if method not in ['pad', 'ffill', 'bfill', None]:
            raise ValueError(f"Parameter 'method' should be 'pad', 'ffill', 'bfill' or None, got {method}")
        return self._qc.replace(self, to_replace, value, inplace, limit, regex, method)

    def iterrows(self):
        iterator = DataFrameIterator(self, 0)
        for v in iterator:
            idx, ser = v
            yield (idx, mpd.Series(ser))

    def get_object_ref(self):
        """Return object id in list when using shared memory backends. Partition shape is not preserved in the
        result."""
        if config.get_concurrency_mode() not in ['yr']:
            warnings.warn("Warning: Can only call get_object_id when using yr backend.")
            return None

        id_list = []
        for row in self.backend_frame.partitions:
            for part in row:
                id_list.append(part.data_id)
        return id_list

    def repartition(self, shape):
        """Perform repartition on current dataframe.

        Args:
            shape(tuple): The expected output partition shape on (row, column) axis.

        Raises:
            TypeError: When parameter shape is not a tuple.
            ValueError: When parameter shape is not a tuple of length 2.
        """
        if not isinstance(shape, tuple):
            raise TypeError(f"Expected a tuple, got {type(shape)}")
        if len(shape) != 2:
            raise ValueError(f"Expected a tuple of length 2, got {len(shape)}")
        partition_shape = self.backend_frame.partition_shape
        new_partition_shape = list(shape)
        if shape[0] == -1:
            new_partition_shape[0] = partition_shape[0]
        if shape[1] == -1:
            new_partition_shape[1] = partition_shape[1]
        self.backend_frame = self.backend_frame.repartition(tuple(new_partition_shape))

    def __invert__(self):
        output_dataframe = self._qc.invert(self)
        return output_dataframe

    def __getattribute__(self, item):
        ## handling the edge case when the input is empty
        attr = super().__getattribute__(item)
        if callable(attr) and self.empty and hasattr(pandas.core.generic.NDFrame, item):
            def default_func(*args, **kwargs):
                return self._qc.default_to_pandas(self, item, *args, **kwargs)

            return default_func
        return attr

    def remote_to_numpy(self):
        return self._qc.remote_to_numpy(self)
