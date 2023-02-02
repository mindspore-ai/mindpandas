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
Series: is one-dimensional ndarray with axis labels.
"""

import warnings

import numpy as np
import pandas
from pandas._libs.lib import no_default, is_scalar
from pandas.api.types import is_bool
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.indexing import convert_to_index_sliceable
from pandas.util._validators import validate_bool_kwarg

import mindpandas as mpd
from mindpandas.backend.base_frame import BaseFrame
from mindpandas.backend.eager.eager_frame import EagerFrame
from .backend.base_io import BaseIO
from .util import is_full_grab_slice, NO_VALUE
from .iterator import DataFrameIterator


class Series:
    """
    This class is used to process series operations.
    """
    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False):
        super(Series, self).__init__()

        if data is None:
            self._name = name
        else:
            if isinstance(data, BaseFrame):
                self._name = name
                self.backend_frame = data
            else:
                if isinstance(data, mpd.Series):
                    data = data.to_pandas()
                if name is None:
                    if isinstance(data, EagerFrame):
                        name = data.columns if data.columns else "__unsqueeze_series__"
                    elif isinstance(data, pandas.Series):
                        name = data.name if data.name else "__unsqueeze_series__"
                    else:
                        name = "__unsqueeze_series__"
                self._name = name
                self.backend_frame = BaseIO.build_series_backend_frame(
                    data, index, dtype, name, copy)

        from .compiler.query_compiler import QueryCompiler as qc
        self._qc = qc
        if data is None:
            self.backend_frame = BaseFrame.create()

        # flag to determine if series is from a dataframe
        self.parent = None
        # if parent_axis==0: series comes from the row via df.loc[row]
        # if parent_axis==1: series comes from the column via df[column]
        self.parent_axis = 0
        self.parent_key = None

    @property
    def name(self):
        """The name of the series."""
        return self._name

    @property
    def loc(self):
        """Access a group of rows and columns by labels or a boolean array."""
        from .index import _Loc
        return _Loc(self)

    @property
    def iloc(self):
        """Access a group of rows and columns by integer-location based indexing."""
        from .index import _ILoc
        return _ILoc(self)

    def head(self, n=5):
        """Return the first n rows."""
        return self.iloc[:n]

    def tail(self, n=5):
        """Return the last n rows."""
        if n == 0:
            return self.iloc[self.backend_frame.num_rows:]
        return self.iloc[-n:]

    @name.setter
    def name(self, value):
        """The setter method of property 'name'.

        Args:
            value: The value to be set as the name of the series.

        Raises:
            ValueError: When '__unsqueeze_series__' is passed as value,
                because it's a reserved keyword.
        """
        if value == '__unsqueeze_series__':
            raise ValueError(
                "'__unsqueeze_series__' is a reserved keyword, please use another name.")

        self._qc.set_series_name(self, value)
        self._name = value

    def __repr__(self):
        """Print partitions for debugging."""
        return self.backend_frame.to_pandas(force_series=True).__repr__()

    def __len__(self):
        """Get the length of the series."""
        return len(self.index)

    def to_numpy(self, dtype=None, copy=False, na_value=no_default):
        """Convert the series to numpy."""
        return self.backend_frame.to_numpy(dtype, copy, na_value).flatten()

    def to_pandas(self):
        """Convert the series to pandas."""
        return self.backend_frame.to_pandas(force_series=True)

    def to_frame(self, name=None):
        """Convert Series to Dataframe.

        Args:
            name: The column name of the converted dataframe.

        Returns:
            A mindpandas.DataFrame.
        """
        return self._qc.to_frame(self, name=name)

    @property
    def dtype(self):
        """Return the dtype object of the underlying data.
        """
        return self.dtypes

    @property
    def dtypes(self):
        """Return the dtype object of the underlying data.
        """
        return self._qc.dtypes(input_dataframe=self)

    @property
    def _index(self):
        """Returns the index (axis labels) of the series. Not called by external API."""
        return self.backend_frame.index

    @property
    def index(self):
        """Returns the index (axis labels) of the series."""
        return self._index

    @index.setter
    def index(self, new_index):
        """Sets index of series."""
        self.set_axis(labels=new_index, axis=0, inplace=True)

    def _validate_set_axis(self, axis, new_labels):
        """Verifies that the labels to set along axis is valid."""
        if not is_list_like(new_labels):
            raise ValueError(f"Index(...) must be called with a collection of some kind, "
                             f"'%s' was passed." % new_labels)
        label_len = len(new_labels)
        if axis in [0, 'index']:
            row_nums = self.backend_frame.num_rows
            if not label_len == row_nums:
                raise ValueError(f"Length mismatch: Expected axis has {row_nums} elements, "
                                 f"new values have {label_len} elements.")

    def set_axis(self, labels, axis=0, inplace=False):
        """Sets labels to the given axis."""
        if axis not in [0, 'index']:
            raise ValueError(f" No axis named {axis} for object type Series")
        self._validate_set_axis(axis=axis, new_labels=labels)
        series = self._qc.set_axis(input_dataframe=self, labels=labels, axis=axis, inplace=inplace)
        if inplace:
            self.backend_frame = series.backend_frame
            return None
        return series

    @property
    def shape(self):
        """Returns shape of series."""
        return (len(self.index),)

    @property
    def size(self):
        """
        Return the number of elements in the underlying data.
        """
        return len(self.backend_frame.index) * len(self.backend_frame.columns)

    @property
    def empty(self):
        """Returns whether the series is empty."""
        return len(self.index) == 0

    @property
    def values(self):
        """Returns the values in the series."""
        return self.to_numpy()

    def groupby(
            self,
            by=None,
            axis=0,
            level=None,
            as_index=True,
            sort=True,
            group_keys=True,
            squeeze: bool = no_default,
            observed=False,
            dropna=True,
    ):
        """Groupby on Series."""
        if squeeze is not no_default:
            warnings.warn(
                (
                    "The `squeeze` parameter is deprecated in pandas 1.1.0."
                ),
                FutureWarning,
                stacklevel=2,
            )
        else:
            squeeze = False

        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")

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

    def rolling(
            self,
            window=None,
            min_periods=None,
            center=False,
            win_type=None,
            on=None,
            axis=0,
            closed=None,
            method="single"

    ):
        """Perform Series rolling window operations."""
        return self._qc.rolling(
            input_dataframe=self,
            window=window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
            method=method
        )

    def squeeze(self, axis=None):
        """Squeeze series by axis."""
        import copy
        axis = self._get_axis_number(axis) if axis is not None else None
        if len(self.index) == 1:
            return self.to_pandas().squeeze()
        return copy.copy(self)

    def _stat_op(self, op_name, **kwargs):
        """
        Do common statistic reduce operations under frame.
        Args:
            op_name : str. Name of method to apply.
            **kwargs : dict. Additional keyword arguments to pass to `op_name`.
        Returns:
            scalar, Series or DataFrame
        """
        if op_name not in {"max", "min", "median", "mean", "count", "sum", "std", "var", "prod"}:
            raise NotImplementedError("Operation not supported")

        numeric_only = kwargs.get("numeric_only", NO_VALUE)
        if numeric_only is True:
            raise NotImplementedError(f"Series.{op_name} does not implement numeric_only.")

        axis = kwargs.get("axis", NO_VALUE)
        if axis is not NO_VALUE:
            axis = self._get_axis_number(axis)
            kwargs["axis"] = axis

        level = kwargs.get("level", NO_VALUE)
        if level is not NO_VALUE and level is not None:
            return self._qc.default_to_pandas(df=self, df_method=op_name, force_series=True, **kwargs)

        return self._qc.stat_op(self, op_name=op_name, **kwargs)

    def max(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        return self._stat_op("max", axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)

    def min(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        return self._stat_op("min", axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)

    def median(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        return self._stat_op("median", axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)

    def mean(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        if not is_numeric_dtype(self.dtype):
            raise TypeError("unsupported operand type")
        return self._stat_op("mean", axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)

    def count(self, level=None):
        return self._stat_op("count", level=level)

    def std(self, axis=None, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
        return self._stat_op("std", axis=axis, skipna=skipna, level=level, ddof=ddof, numeric_only=numeric_only,
                             **kwargs)

    def var(self, axis=None, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
        return self._stat_op("var", axis=axis, skipna=skipna, level=level, ddof=ddof, numeric_only=numeric_only,
                             **kwargs)

    def sum(self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0, **kwargs):
        return self._stat_op("sum", axis=axis, skipna=skipna, level=level, numeric_only=numeric_only,
                             min_count=min_count, **kwargs)

    def prod(self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0, **kwargs):
        return self._stat_op("prod", axis=axis, skipna=skipna, level=level, numeric_only=numeric_only,
                             min_count=min_count, **kwargs)

    product = prod

    def _logical_op(self, op_name, axis, bool_only=None, **kwargs):
        if op_name not in {"all", "any"}:
            raise NotImplementedError("Operation not supported")
        axis = self._get_axis_number(axis)
        if bool_only is not None and bool_only is not False:
            raise NotImplementedError(f"Series.{op_name} does not implement numeric_only.")
        return self._qc.logical_op(self, op_name=op_name, axis=axis, bool_only=bool_only, **kwargs)

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        return self._logical_op("all", axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        return self._logical_op("any", axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

    def _get_axis_number(self, axis):
        """Get axis number, 0 or 1."""
        if axis is no_default:
            axis = None
        return pandas.Series._get_axis_number(axis) if axis is not None else 0

    def __iter__(self):
        """Get iterable over the series."""
        return self.to_pandas().__iter__()

    def __setitem__(self, key, value):
        """Setitem for series."""
        if self.parent is not None:
            index_col = self.parent.columns.tolist().index(self.parent_key[0])
            if not is_scalar(key):
                index_row = [self.parent.index.tolist().index(index) for index in key]
            else:
                index_row = [self.parent.index.tolist().index(key)]
            self._qc.setitem_elements(
                self.parent, 'iloc', [index_col], index_row, value)
        else:
            if isinstance(key, slice):
                indexer = convert_to_index_sliceable(pandas.DataFrame(index=self.index), key)
                self.iloc[indexer] = value
            else:
                if isinstance(key, int):
                    self.iloc[key] = value
                else:
                    self.loc[key] = value

    def copy(self, deep=True):
        """Returns a copy of the series."""
        return self._qc.copy(self, deep=deep)

    def __getitem__(self, key):
        """Getitem for series."""
        indexer = None
        if isinstance(key, mpd.Series):
            return self._qc.getitem_array(self, key)
        if isinstance(key, (slice, str)):
            indexer = convert_to_index_sliceable(pandas.DataFrame(index=self.index), key)
        # if key can be convert to slice, call self.iloc directly
        if indexer is not None:
            if is_full_grab_slice(indexer, sequence_len=len(self)):
                return self.copy()
            return self.iloc[indexer]
        key = apply_if_callable(key, self)
        if is_bool_indexer(key) or(is_list_like(key) and all(map(is_bool, key))):
            return self._qc.getitem_row_array(self,
                                              pandas.RangeIndex(len(self.index))[key],
                                              indices_numeric=True)
        if isinstance(key, tuple):
            return self._qc.default_to_pandas(self,
                                              pandas.Series.__getitem__,
                                              key,
                                              force_series=True)
        if not is_list_like(key):
            need_squeeze = True
            key = np.array([key])
        else:
            need_squeeze = False
        try:
            if all(k in self.index for k in key):
                result = self._qc.getitem_row_array(self,
                                                    self.index.get_indexer_for(key),
                                                    indices_numeric=True)
            else:
                result = self._qc.getitem_row_array(self, key, indices_numeric=True)
        except TypeError:
            result = self._qc.getitem_row_array(self, key)
        # squeeze mpd.Series to scalar
        if need_squeeze:
            return result.squeeze()
        return result

    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        """Apply function onto series."""
        outer_kwargs = dict()
        outer_kwargs['func'] = func
        outer_kwargs['convert_dtype'] = convert_dtype
        outer_kwargs['args'] = args
        outer_kwargs['axis'] = 1
        outer_kwargs.update(kwargs)

        result = self._qc.apply(self, **outer_kwargs)
        return result

    def _comp_op(self, func, other, level=None, fill_value=None, axis=0):
        """
        Comparison operators, i.e., ==, <, <=, >, >=, and !=
        :param input: Series
        :param func: the input comparison operator
        :param other: Series or DataFrame or scalar value (currently only value is supported.)
        :param level: only None is supported
        :param fill_value: only None is supported
        :param axis: 0 or index (only index axis is supported)
        :return: Series of Boolean type of the same shape as the input
        """
        if func in {'__eq__', '__le__', '__lt__', '__ge__', '__gt__', '__ne__'}:
            if isinstance(other, (Series, pandas.Series)) and self.shape != other.shape:
                raise ValueError("Can only compare identically-labeled Series objects")
            if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
                raise ValueError(f"('Lengths must match to compare', ({len(self)},), ({len(other)},))")
            func = func.strip('_')
        elif func in {'eq', 'le', 'lt', 'ge', 'gt', 'ne'}:
            if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
                raise ValueError("Lengths must be equal")
        elif func not in {'equals'}:
            raise NotImplementedError(f"{func} operation is not supported yet")

        axis = self._get_axis_number(axis)

        if level is not None or fill_value is not None:
            return self._qc.default_to_pandas(df=self, df_method=func, other=other, level=level, fill_value=fill_value,
                                              axis=0, force_series=True)
        if is_scalar(other):
            return self._qc.series_comp_op(self, func, other, True, level, fill_value, axis)

        if isinstance(other, (list, np.ndarray, tuple)):
            if not pandas.Index(np.arange(0, len(self))).equals(self.index):
                self.reset_index(drop=True, inplace=True)
            other_series = Series(other, index=self.index[:len(self)])
            return self._qc.series_comp_op(self, func, other_series, False, level, fill_value, axis)

        if isinstance(other, pandas.Series):
            other = Series(other)

        if isinstance(other, Series):
            result = self._qc.series_comp_op(self, func, other, False, level, fill_value, axis)
            if 'mixed' in result.index.inferred_type or self.index.equals(other.index):
                return result
            return result.sort_index()

        # other situations
        return self._qc.default_to_pandas(df=self, df_method=func, other=other, level=level, fill_value=fill_value,
                                          axis=0, force_series=True)

    def eq(self, other, level=None, fill_value=None, axis=0):
        """Elementwise equality comparison for series."""
        return self._comp_op('eq', other, level, fill_value, axis)

    def __eq__(self, other):
        """Elementwise equality comparison for series."""
        return self._comp_op('__eq__', other, None, None, 0)

    def le(self, other, level=None, fill_value=None, axis=0):
        """Elementwise <= comparison for series."""
        return self._comp_op('le', other, level, fill_value, axis)

    def __le__(self, other):
        """Elementwise <= comparison for series."""
        return self._comp_op('__le__', other, None, None, 0)

    def lt(self, other, level=None, fill_value=None, axis=0):
        """Elementwise < comparison for series."""
        return self._comp_op('lt', other, level, fill_value, axis)

    def __lt__(self, other):
        """Elementwise < comparison for series."""
        return self._comp_op('__lt__', other, None, None, 0)

    def ge(self, other, level=None, fill_value=None, axis=0):
        """Elementwise >= comparison for series."""
        return self._comp_op('ge', other, level, fill_value, axis)

    def __ge__(self, other):
        """Elementwise >= comparison for series."""
        return self._comp_op('__ge__', other, None, None, 0)

    def gt(self, other, level=None, fill_value=None, axis=0):
        """Elementwise > comparison for series."""
        return self._comp_op('gt', other, level, fill_value, axis)

    def __gt__(self, other):
        """Elementwise > comparison for series."""
        return self._comp_op('__gt__', other, None, None, 0)

    def ne(self, other, level=None, fill_value=None, axis=0):
        """Elementwise != comparison for series."""
        return self._comp_op('ne', other, level, fill_value, axis)

    def __ne__(self, other):
        """Elementwise != comparison for series."""
        return self._comp_op('__ne__', other, None, None, 0)

    def equals(self, other):
        """Returns true if the series and other are equal."""
        # check Series shapes and dtypes in advanced
        if self.shape != other.shape or self.dtypes != other.dtypes:
            return False
        equal = self._comp_op('equals', other, None, None, 0).all(axis=0)
        # Convert numpy.bool to python bool
        return bool(equal)

    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
        """Returns a count of each unique value in the series."""
        counted_values = self._qc.series_value_counts(
            data=self,
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            bins=bins,
            dropna=dropna,
        )
        return counted_values

    def tolist(self):
        """Returns series as a list."""
        return self._qc.default_to_pandas(self, self.tolist, force_series=True)

    def to_list(self):
        """Returns series as a list."""
        return self._qc.default_to_pandas(self, self.to_list, force_series=True)

    def to_dict(self, into=dict):
        """Returns series as a dict."""
        return self._qc.default_to_pandas(self, self.to_dict, into=into, force_series=True)

    def sort_index(self,
                   axis=0,
                   level=None,
                   ascending=True,
                   inplace=False,
                   kind='quicksort',
                   na_position='last',
                   sort_remaining=True,
                   ignore_index=False,
                   key=None):
        """Returns series sorted by index."""
        sorted_series = self._qc.default_to_pandas(df=self,
                                                   df_method=self.sort_index,
                                                   axis=axis,
                                                   level=level,
                                                   ascending=ascending,
                                                   inplace=inplace,
                                                   kind=kind,
                                                   na_position=na_position,
                                                   sort_remaining=sort_remaining,
                                                   ignore_index=ignore_index,
                                                   key=key,
                                                   force_series=True)
        return sorted_series

    def reset_index(self, level=None, drop=False, name=no_default, inplace=False):
        """Reset series index."""
        reseted_series = self._qc.default_to_pandas(df=self,
                                                    df_method=self.reset_index,
                                                    level=level,
                                                    drop=drop,
                                                    name=name,
                                                    inplace=inplace,
                                                    force_series=True)
        return reseted_series

    def abs(self):
        return self._qc.abs(self)

    __abs__ = abs

    def fillna(self,
               value=None,
               method=None,
               axis=None,
               inplace=False,
               limit=None,
               downcast=None):
        """Fill Nan values in series."""
        ms_inplace = validate_bool_kwarg(inplace, "inplace")
        ms_axis = self._get_axis_number(axis)
        ms_method = method
        ms_value = value
        if isinstance(ms_value, (list, tuple)):
            raise TypeError(
                ' "value" parameter must be a scalar or dict, but '
                'you passed a "{0}"'.format(type(ms_value).__name__)
            )
        # if ms_value is None and method is None, raise error message
        if ms_value is None and ms_method is None:
            raise ValueError("In Mindpandas, must specify a fill method or value")
        # check limit value
        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("limit must be an integer")
            if limit <= 0:
                raise ValueError("limit must be greater than 0")
        # check if ms_value is instance of Series
        if isinstance(ms_value, Series):
            ms_value = ms_value.to_pandas()
        # raise error message if ms_value is not None and method is not None
        if ms_value is not None and ms_method is not None:
            raise ValueError("In Mindpandas, cannot specify both a fill method or value")
        # raise error if method is not valid
        if ms_method is not None and ms_method not in ["backfill", "bfill", "pad", "ffill"]:
            expecting = "pad (fill) or backfill (bfill)"
            msg = "Invalid fill method. Expecting {expecting}. Got {method}".format(
                expecting=expecting, method=ms_method
            )
            raise ValueError(msg)

        output_dataframe = self._qc.fillna(input_dataframe=self, squeeze_self=True,
                                           value=ms_value, method=ms_method, axis=ms_axis,
                                           inplace=False, limit=limit, downcast=downcast,
                                           is_series=True)
        if ms_inplace:
            self.backend_frame = output_dataframe.backend_frame
            return None
        return output_dataframe

    def isna(self):
        """Returns series indicating whether element is Nan."""
        output_series = self._qc.isna(input_dataframe=self, is_series=True)
        return output_series

    def isnull(self):
        """Returns series indicating whether element is Nan."""
        return self.isna()

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
            raise ValueError(
                f"Value of errors should be in ['ignore', 'raise'], got {errors}")
        if isinstance(dtype, dict):
            for col_name in dtype.keys():
                if col_name != self.name:
                    raise KeyError(
                        "Only Series name can be used as key in Series dtype mappings."
                    )

        return self._qc.astype(self, dtype=dtype, copy=copy, errors=errors)

    def add(self, other, level=None, fill_value=None, axis=0):
        """Add operation for series."""
        return self._math_op("add", other, axis, level, fill_value)

    def sub(self, other, level=None, fill_value=None, axis=0):
        """Subtract operation for series."""
        return self._math_op("sub", other, axis, level, fill_value)

    def mul(self, other, level=None, fill_value=None, axis=0):
        """Multiply operation for series."""
        return self._math_op("mul", other, axis, level, fill_value)

    def div(self, other, level=None, fill_value=None, axis=0):
        """Divide operation for series."""
        return self._math_op("div", other, axis, level, fill_value)

    def truediv(self, other, level=None, fill_value=None, axis=0):
        """Floating point divide operation for series."""
        return self._math_op("truediv", other, axis, level, fill_value)

    def floordiv(self, other, level=None, fill_value=None, axis=0):
        """Floor division operation for series."""
        return self._math_op("floordiv", other, axis, level, fill_value)

    def mod(self, other, level=None, fill_value=None, axis=0):
        """Mod operation for series."""
        return self._math_op("mod", other, axis, level, fill_value)

    def pow(self, other, level=None, fill_value=None, axis=0):
        """power operation for series."""
        return self._math_op("pow", other, axis, level, fill_value)

    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __truediv__ = truediv
    __floordiv__ = floordiv
    __mod__ = mod
    __pow__ = pow

    def _math_op(self, op, other, axis=0, level=None, fill_value=None):
        """ The wrapper of the general math operator
        Args:
            op: string. The math operator name.
            other: a mindpandas.Series, pandas.Series or scalar value.
            axis: axis is 0 for the Series math operator.
            level: int or name. Broadcast across a level, matching Index values on the passed
                MultiIndex level.
            fill_value: None or float value, default None (NaN).
                Fill existing missing (NaN) values, and any new element needed for successful
                Series alignment, with this value before computation. If data in both
                corresponding Series locations is missing the result of filling (at that location)
                will be missing.

        Returns:
            A mindpandas.Series.
        """
        if axis is not None:
            axis = self._get_axis_number(axis)
            if axis != 0:
                raise ValueError(f"No axis named {axis} for object type Series")

        if op not in ["add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow"]:
            raise NotImplementedError(f"{op} operation is not supported yet")

        if level is None:
            if is_scalar(other):
                return self._qc.math_op(self, op, other, axis, level, fill_value)

            if isinstance(other, (mpd.Series, pandas.Series)):
                result = self._qc.math_op(self, op, other, axis, level, fill_value)
                if 'mixed' in result.index.inferred_type or self.index.equals(other.index):
                    return result
                return result.sort_index()

        # Other situations use pandas for better performance
        return self._qc.default_to_pandas(df=self,
                                          df_method=op,
                                          other=other,
                                          axis=axis,
                                          level=level,
                                          fill_value=fill_value,
                                          force_series=True)

    def _cum_op(self, method, axis=None, skipna=True, **kwargs):
        axis = self._get_axis_number(axis)
        if not isinstance(skipna, bool):
            raise TypeError(f"For {method} operation, 'skipna' should be a bool, got {type(skipna)}.")
        output = self._qc.cum_op(self, method=method, axis=axis, skipna=skipna, **kwargs)
        return output

    def cumsum(self, axis=None, skipna=True, **kwargs):
        return self._cum_op(method='cumsum', axis=axis, skipna=skipna, **kwargs)

    def cummin(self, axis=None, skipna=True, **kwargs):
        return self._cum_op(method='cummin', axis=axis, skipna=skipna, **kwargs)

    def cummax(self, axis=None, skipna=True, **kwargs):
        return self._cum_op(method='cummax', axis=axis, skipna=skipna, **kwargs)

    def cumprod(self, axis=None, skipna=True, **kwargs):
        return self._cum_op(method='cumprod', axis=axis, skipna=skipna, **kwargs)

    _comp_op_name_mapping = {'eq': '__eq__', 'le': '__le__', 'lt': '__lt__',
                             'ge': '__ge__', 'gt': '__gt__', 'ne': '__ne__'}

    def memory_usage(self, index=True, deep=False):
        """
        Return the memory usage of each column in bytes.
        """
        if index:
            result = self._qc.memory_usage(self, index=False, deep=deep)
            index_value = self.index.memory_usage(deep=deep)
            return (result.sum() + index_value).item()
        return (self._qc.memory_usage(self, index=index, deep=deep).sum()).item()

    def __getattribute__(self, item):
        """Getattribute for series."""
        ## handling the edge case when the input is empty
        attr = super().__getattribute__(item)
        if callable(attr) and self.empty:
            if hasattr(pandas.core.generic.NDFrame, item):
                def default_func(*args, **kwargs):
                    return self._qc.default_to_pandas(self, item, *args, force_series=True, **kwargs)
                return default_func
        return attr

    def item(self):
        """Return the first element of the underlying data as a Python scalar."""
        if len(self) != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar.")
        obj = self[0]
        if hasattr(obj, 'dtype'):
            obj = obj.item()
        return obj

    def items(self):
        """Lazily iterate over (index, value) tuples."""
        df_iterator = DataFrameIterator(self.to_frame(), 0)
        for t in df_iterator:
            yield t[0], t[1].squeeze()

    def isin(self, values):
        """Whether elements in Series are contained in value."""
        if isinstance(values, (mpd.DataFrame, mpd.Series)):
            values = values.to_pandas()
        output = self._qc.isin(input_dataframe=self, values=values)
        return output
