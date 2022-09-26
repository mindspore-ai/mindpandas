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
from pandas.core.dtypes.common import is_list_like
from pandas.core.indexing import convert_to_index_sliceable
from pandas.util._validators import validate_bool_kwarg

import mindpandas as mpd
from mindpandas.backend.base_frame import BaseFrame
from mindpandas.backend.eager.eager_frame import EagerFrame
from .backend.base_io import BaseIO
from .util import is_full_grab_slice


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
        return self._qc.set_axis(input_dataframe=self, labels=labels, axis=axis, inplace=inplace)

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

    def _statistic_operation(self,
                             op_name,
                             axis=None,
                             skipna=True,
                             level=None,
                             numeric_only=None,
                             **kwargs):
        """
        Do common statistic reduce operations under Series.
        Args:
            op_name : str. Name of method to apply.
            axis : int or str. Axis to apply method on.
            skipna : bool. Exclude NA/null values when computing the result.
            level : int or str. If specified `axis` is a MultiIndex, applying method
                along a particular level, collapsing into a Series.
            numeric_only : bool, optional. Not implemented for mindspore.Series
            **kwargs : dict. Additional keyword arguments to pass to `op_name`.
        Returns:
            scalar, Series
        """
        axis = self._get_axis_number(axis)

        if level is not None:
            return self._qc.default_to_pandas(df=self,
                                              df_method=op_name,
                                              axis=axis,
                                              skipna=skipna,
                                              level=level,
                                              numeric_only=numeric_only,
                                              force_series=True,
                                              **kwargs)

        if numeric_only is not None:
            raise NotImplementedError(
                "numeric_only is not implemented for mindspore.Series")

        output_dataframe = getattr(self._qc, op_name)(self,
                                                      axis,
                                                      skipna,
                                                      level,
                                                      numeric_only,
                                                      **kwargs)
        return output_dataframe

    def mean(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        """Return mean of series."""
        return self._statistic_operation("mean", axis, skipna, level, numeric_only, **kwargs)

    def median(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        """Return median of series."""
        return self._statistic_operation("median", axis, skipna, level, numeric_only, **kwargs)

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
        comparator_list = ['eq', 'le', 'lt', 'ge', 'gt', 'ne', 'equals']
        if axis is None:
            axis = 0
        axis = self._get_axis_number(axis)
        if axis != 0:
            raise ValueError(f"No axis named {axis} for object type Series")
        if level is not None or fill_value is not None:
            if isinstance(other, mpd.Series):
                other_series = other.to_pandas()
            else:
                other_series = other
            return self._qc.default_to_pandas(df=self,
                                              df_method=func,
                                              other=other_series,
                                              level=level,
                                              fill_value=fill_value,
                                              axis=0,
                                              force_series=True)
        if func in comparator_list:
            # check other is pandas.Series or not
            if isinstance(other, pandas.Series):
                other = Series(other)
            if isinstance(other, (list, np.ndarray, pandas.Series, tuple)) and not \
               pandas.Index(np.arange(0, len(self))).equals(self.index):
                self.reset_index(drop=True, inplace=True)
            if isinstance(other, (list, np.ndarray, tuple)):
                other_series = Series(other, index=self.index[:len(self)])
                return self._qc.series_comp_op(self,
                                               func,
                                               other_series,
                                               False,
                                               level,
                                               fill_value,
                                               axis,
                                               sort=False)
            if isinstance(other, Series):
                return self._qc.series_comp_op(self,
                                               func,
                                               other,
                                               False,
                                               level,
                                               fill_value,
                                               axis,
                                               sort=False)
            if is_scalar(other):
                return self._qc.series_comp_op(self,
                                               func,
                                               other,
                                               True,
                                               level,
                                               fill_value,
                                               axis,
                                               sort=False)
            # handle special case of eq that other is a dict or set
            if isinstance(other, (dict, set)) and func in ('eq', 'ne'):
                return self._qc.default_to_pandas(df=self,
                                                  df_method=func,
                                                  other=other,
                                                  level=level,
                                                  fill_value=fill_value,
                                                  axis=0,
                                                  force_series=True)
            raise TypeError(
                f"Argument other of type {type(other)} is not supported.")
        raise ValueError(f"Argument func {func} is not supported.")

    def eq(self, other, level=None, fill_value=None, axis=0):
        """Elementwise equality comparison for series."""
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError("Lengths must be equal")
        return self._comp_op('eq', other, level, fill_value, axis)

    def __eq__(self, other):
        """Elementwise equality comparison for series."""
        if isinstance(other, (Series, pandas.Series)) and self.shape != other.shape:
            raise ValueError("Can only compare identically-labeled Series objects")
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError(f"('Lengths must match to compare', ({len(self)},), ({len(other)},))")
        return self._comp_op('eq', other, None, None, 0)
        # return self.apply(operator.eq, (other,))

    def le(self, other, level=None, fill_value=None, axis=0):
        """Elementwise <= comparison for series."""
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError("Lengths must be equal")
        return self._comp_op('le', other, level, fill_value, axis)

    def __le__(self, other):
        """Elementwise <= comparison for series."""
        if isinstance(other, (Series, pandas.Series)) and self.shape != other.shape:
            raise ValueError("Can only compare identically-labeled Series objects")
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError(f"('Lengths must match to compare', ({len(self)},), ({len(other)},))")
        return self._comp_op('le', other, None, None, 0)

    def lt(self, other, level=None, fill_value=None, axis=0):
        """Elementwise < comparison for series."""
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError("Lengths must be equal")
        return self._comp_op('lt', other, level, fill_value, axis)

    def __lt__(self, other):
        """Elementwise < comparison for series."""
        if isinstance(other, (Series, pandas.Series)) and self.shape != other.shape:
            raise ValueError("Can only compare identically-labeled Series objects")
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError(f"('Lengths must match to compare', ({len(self)},), ({len(other)},))")
        return self._comp_op('lt', other, None, None, 0)

    def ge(self, other, level=None, fill_value=None, axis=0):
        """Elementwise >= comparison for series."""
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError("Lengths must be equal")
        return self._comp_op('ge', other, level, fill_value, axis)

    def __ge__(self, other):
        """Elementwise >= comparison for series."""
        if isinstance(other, (Series, pandas.Series)) and self.shape != other.shape:
            raise ValueError("Can only compare identically-labeled Series objects")
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError(f"('Lengths must match to compare', ({len(self)},), ({len(other)},))")
        return self._comp_op('ge', other, None, None, 0)

    def gt(self, other, level=None, fill_value=None, axis=0):
        """Elementwise > comparison for series."""
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError("Lengths must be equal")
        return self._comp_op('gt', other, level, fill_value, axis)

    def __gt__(self, other):
        "Elementwise > comparison for series."
        if isinstance(other, (Series, pandas.Series)) and self.shape != other.shape:
            raise ValueError("Can only compare identically-labeled Series objects")
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError(f"('Lengths must match to compare', ({len(self)},), ({len(other)},))")
        return self._comp_op('gt', other, None, None, 0)

    def ne(self, other, level=None, fill_value=None, axis=0):
        """Elementwise != comparison for series."""
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError("Lengths must be equal")
        return self._comp_op('ne', other, level, fill_value, axis)

    def __ne__(self, other):
        """Elementwise != comparison for series."""
        if isinstance(other, (Series, pandas.Series)) and self.shape != other.shape:
            raise ValueError("Can only compare identically-labeled Series objects")
        if isinstance(other, (list, np.ndarray)) and len(self) != len(other):
            raise ValueError(f"('Lengths must match to compare', ({len(self)},), ({len(other)},))")
        return self._comp_op('ne', other, None, None, 0)

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
        # check limit value
        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("limit must be an integer")
            if limit <= 0:
                raise ValueError("limit must be greater than 0")
        output_dataframe = self._qc.fillna(
            input_dataframe=self,
            squeeze_self=True,
            value=ms_value,
            method=ms_method,
            axis=ms_axis,
            inplace=False,
            limit=limit,
            downcast=downcast,
            is_series=True,
        )
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

    def max(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        """Returns max of series."""
        axis = self._get_axis_number(axis)
        result = self._qc.max(self, axis=axis, skipna=skipna,
                              level=level, numeric_only=numeric_only, **kwargs)
        return result

    def min(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        """Returns min of series."""
        axis = self._get_axis_number(axis)
        result = self._qc.min(self, axis=axis, skipna=skipna,
                              level=level, numeric_only=numeric_only, **kwargs)
        return result

    def std(self, axis=None, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
        """Returns standard deviation of series."""
        return self._statistic_operation("std",
                                         axis,
                                         skipna,
                                         level,
                                         numeric_only,
                                         ddof=ddof,
                                         **kwargs)

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """Returns true if all elements are true."""
        axis = self._get_axis_number(axis)
        if bool_only is not None and (not isinstance(bool_only, bool) or bool_only is True):
            raise NotImplementedError("Series.all does not implement numeric_only.")
        return self._qc.all(self,
                            axis=axis,
                            bool_only=bool_only,
                            skipna=skipna,
                            level=level,
                            **kwargs)

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """Returns true if any element is true."""
        axis = self._get_axis_number(axis)
        if bool_only is not None and (not isinstance(bool_only, bool) or bool_only is True):
            raise NotImplementedError(
                "Series.any does not implement numeric_only.")
        return self._qc.any(self,
                            axis=axis,
                            bool_only=bool_only,
                            skipna=skipna,
                            level=level,
                            **kwargs)

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
                return self._qc.math_op(self, op, other, axis, level, fill_value)

        # Other situations use pandas for better performance
        return self._qc.default_to_pandas(df=self,
                                          df_method=op,
                                          other=other,
                                          axis=axis,
                                          level=level,
                                          fill_value=fill_value,
                                          force_series=True)

    def cumsum(self, axis=None, skipna=True, **kwargs):
        """
        Return cumulative sum over a Series axis.

        Args:
            axis: {0 or ‘index’, 1 or ‘columns’} default 0. The index or the name of the axis.
                0 is equivalent to None or ‘index’.
            skipna: bool, default True. Exclude NA/null values. If an entire row/column is NA,
                the result will be NA.
            *args, **kwargs: Additional keywords have no effect but might be accepted for
                compatibility with NumPy.

        Returns:
            scalar or Series. Return cumulative sum of scalar or Series.

        Supportes Platforms:
            ``CPU``
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        output_dataframe = self._qc.cumsum(self, axis=axis, skipna=skipna, **kwargs)
        return output_dataframe

    def sum(self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0, **kwargs):
        """Returns sum of the values over the specified axis."""
        axis = self._get_axis_number(axis)

        if level is not None:
            return self._qc.default_to_pandas(df=self,
                                              df_method=self.sum,
                                              axis=axis,
                                              skipna=skipna,
                                              level=level,
                                              numeric_only=numeric_only,
                                              force_series=True,
                                              min_count=min_count,
                                              **kwargs)

        if numeric_only is None:
            numeric_only = True

        output_dataframe = self._qc.sum(self,
                                        axis=axis,
                                        skipna=skipna,
                                        numeric_only=numeric_only,
                                        min_count=min_count,
                                        **kwargs)
        return output_dataframe

    _comp_op_name_mapping = {'eq': '__eq__', 'le': '__le__', 'lt': '__lt__',
                             'ge': '__ge__', 'gt': '__gt__', 'ne': '__ne__'}

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
