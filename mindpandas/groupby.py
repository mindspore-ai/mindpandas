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
This module defines the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces.
"""
import numpy as np
import pandas

import mindpandas as mpd
from mindpandas.compiler.function_factory import FunctionFactory as ff
from mindpandas.dataframe import (DataFrame, hashable, is_list_like,
                                  no_default)


class DataFrameGroupBy:
    """
    A class that hold the groupby interfaces for DataFrame
    """

    def __init__(self,
                 dataframe,
                 by=None,
                 axis=0,
                 level=None,
                 as_index=True,
                 sort=True,
                 group_keys=True,
                 squeeze=no_default,  # Deprecated argument since pandas version 1.1.0.
                 observed=False,
                 dropna=True,
                 by_names=None,
                 by_dataframe=None,
                 drop=False,
                 is_mapping=False,
                 default_to_pandas=False,
                 get_item_key=None,
                 **kwargs
                 ):
        super(DataFrameGroupBy, self).__init__()
        self._df = dataframe
        self.axis = axis
        self.by = by
        self.level = level
        self.squeeze = squeeze
        from .compiler.query_compiler import QueryCompiler as qc
        self._qc = qc
        self.backend_frame = dataframe.backend_frame
        self.columns = self.backend_frame.columns
        self.dtypes = self.backend_frame.dtypes
        self.by_names = by_names
        self.by_dataframe = by_dataframe
        self.drop = drop
        self.default_to_pandas = default_to_pandas
        self._is_series = isinstance(self, SeriesGroupBy)
        self.is_mapping = is_mapping
        self.get_item_key = get_item_key

        self.kwargs = {
            "level": level,
            "as_index": as_index,
            "sort": sort,
            "group_keys": group_keys,
            "observed": observed,
            "dropna": dropna,
            "axis": axis
        }
        self.kwargs.update(kwargs)

    _index_grouped_cache = no_default

    def _attribute_wrapper(self, attribute_name):
        """
        A helper function for cache and get the attribute from the groupby object

        Parameters
        ----------
        attribute_name: str
            The attribute name which want to get from the groupby object
        """
        mask_func = ff.mask_iloc()
        if self._index_grouped_cache is None or no_default:
            index_ = self.backend_frame.index
            def compute_by():
                by = None
                level = []
                if self.level is not None:
                    level = self.level
                    if not isinstance(level, list):
                        level = [level]
                elif isinstance(self.by, list):
                    by = []
                    for obj in self.by:
                        if hashable(obj) and obj in index_:
                            level.append(obj)
                        else:
                            by.append(obj)
                else:
                    by = self.by

                if isinstance(self.by, type(DataFrame)):
                    by = self.by.to_pandas().squeeze().values
                elif by is None:
                    index = index_.name if self.axis == 0 else self.columns.name
                    index = [index] if not isinstance(index, list) else index
                    by = [
                        name
                        for i, name in enumerate(index)
                        if name in level or i in level
                    ]
                else:
                    by = self.by

                return by

            def get_axes():
                return self.columns if self.axis == 0 else index_

            by = compute_by()
            # `dropna` param is the only one that matters for the group indices result
            dropna = self.kwargs.get("dropna", True)
            index = get_axes()
            by_is_list = isinstance(by, list)

            def get_axes_label_case():
                if self.axis == 0:
                    return self.backend_frame.get_columns(by, func=mask_func).to_pandas()
                return self.backend_frame.get_rows(by, func=mask_func).to_pandas()

            def get_axes_list_mapping_case():
                if self.axis == 0:
                    return index_.to_series()
                return self.columns.to_series()

            if ((isinstance(by, list)
                 and all((isinstance(b, (dict, mpd.Series, pandas.Series)) for b in by)))
                    or isinstance(by, (mpd.Series, pandas.Series))):
                self._index_grouped_cache = self._qc.groupby_default_to_pandas(
                    self, attribute_name, force_series=self._is_series)
            elif isinstance(by, dict):
                axis_labels = index_.to_series()
                grouby_obj = axis_labels.groupby(by, dropna=dropna)
                self._index_grouped_cache = getattr(grouby_obj, attribute_name)
            elif ((by_is_list and all(b in index for b in by)) or
                  (not isinstance(by, list) and by in index)):
                axis_labels = get_axes_label_case()
                grouby_obj = axis_labels.groupby(by, dropna=dropna)
                self._index_grouped_cache = getattr(grouby_obj, attribute_name)
            elif ((by_is_list and all(b in index_.names for b in by))
                  or (not by_is_list and by in index_.names)):
                axis_labels = index_.to_series()
                grouby_obj = axis_labels.groupby(by, dropna=dropna)
                self._index_grouped_cache = getattr(grouby_obj, attribute_name)
            elif by_is_list:  # list mapping
                axis_labels = get_axes_list_mapping_case()
                grouby_obj = axis_labels.groupby(by, dropna=dropna)
                self._index_grouped_cache = getattr(grouby_obj, attribute_name)
            else:
                self._index_grouped_cache = self._qc.groupby_default_to_pandas(
                    self, attribute_name, force_series=self._is_series)

        return self._index_grouped_cache

    def _validate_dtypes(self):
        for d in self.dtypes:
            if d.type is np.object_:
                return True
        return False

    @property
    def _index_grouped(self):
        return self._attribute_wrapper("groups")

    @property
    def ngroups(self):
        return len(self)

    @property
    def groups(self):
        return self._index_grouped

    _indices_cache = no_default

    @property
    def indices(self):
        if self._indices_cache is not no_default:
            return self._indices_cache

        self._indices_cache = self._attribute_wrapper("indices")
        return self._indices_cache

    @property
    def ndim(self):
        return 2

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self._iter.__iter__()

    @property
    def _iter(self):
        """
        Helper function for __iter__().
        """
        indices = self.indices
        group_ids = indices.keys()
        if self.axis == 0:
            return (
                (k, self._qc.getitem_row_array(self._df, indexes=indices[k])) for
                k in (sorted(group_ids) if self.kwargs['sort'] else group_ids)
            )
        return (
            (k, self._qc.getitem_column_array(self._df, columns=indices[k],
                                              is_series=len(indices[k]) == 1)) for
            k in (sorted(group_ids) if self.kwargs['sort'] else group_ids)
        )

    def _getitem_key_validation_check(self, key):
        if isinstance(key, (list, tuple, pandas.Series, mpd.Series, pandas.Index, np.ndarray)):
            if len(self.columns.intersection(key)) != len(set(key)):
                error_keys = list(set(key).difference(self.columns))
                raise KeyError(f"Columns not found: {str(error_keys)[1:-1]}")
        elif isinstance(key, (pandas.DataFrame, mpd.DataFrame, dict)):
            raise TypeError(f"unhashable type: '{key.__class__.__name__}'")
        else:
            if key not in self.columns:
                raise KeyError(f"Column not found: {key}")

    def __getitem__(self, key):
        kwargs = {
            **self.kwargs.copy(),
            "by": self.by,
            "axis": self.axis,
            "by_names": self.by_names,
            "by_dataframe": self.by_dataframe
        }
        mask_func = ff.mask_iloc()

        self._getitem_key_validation_check(key)

        if self._is_series and len(key) > 1:
            raise AttributeError("'Series' object has no attribute 'columns'")

        def _is_dataframe(key):
            as_index = kwargs.get("as_index")

            if is_list_like(key):
                is_dataframe = True
            else:
                is_dataframe = False
                if not as_index:
                    is_dataframe = True
                    key = [key]
            return is_dataframe, key

        is_dataframe, key = _is_dataframe(key)

        # NOTE: currently support single or multi column in the DataFrameGroupBy results
        if any([isinstance(key, list),
                isinstance(key, tuple),
                isinstance(key, pandas.Series),
                isinstance(key, mpd.Series)]) or not is_dataframe:

            get_item_key = key
            if isinstance(key, tuple):
                key = list(key)

            if not is_dataframe:
                key = [key]
            # NOTE: kwargs["by_names"] set the indexes for dataframe,
            if isinstance(self.by, list) and all(hashable(b) for b in self.by):   # list of labels
                if isinstance(key, pandas.Series):
                    key = key.tolist()
                elif isinstance(key, mpd.Series):
                    key = key.to_pandas().to_list()

                # Sorted by the same type
                cols_needed = sorted(key+self.by, key=str)
                key = [col for col in self.columns if col in cols_needed]
            elif hashable(self.by):    # a single labels
                cols_needed = sorted(key+[self.by])
                key = [col for col in self.columns if col in cols_needed]
            else:   # _by is mapping
                pass

            new_dataframe = DataFrame(self.backend_frame.get_columns(key, func=mask_func))

            if is_dataframe:
                groupby_obj = DataFrameGroupBy
                drop = self.drop
            else:
                groupby_obj = SeriesGroupBy
                drop = False

            return groupby_obj(
                new_dataframe,
                drop=drop,
                get_item_key=get_item_key,
                **kwargs,
            )

        raise NotImplementedError(
            f"Current DataFrame groupby doesn't support {type(key)} type keys.")

    def apply(self, func, **kwargs):
        """ Apply func group-wise and combine the results together.

            Note: This function is only used for demo purposes.

            TODO: This function requires parallel optimization
        """
        kwargs["func"] = func
        output_dataframe = self._qc.groupby_default_to_pandas(
            self, "apply", force_series=self._is_series, **kwargs)
        return output_dataframe

    def func_wrapper(self, groupby_method_name, **kwargs):
        if isinstance(self.by, pandas.Grouper):
            output_dataframe = self._qc.groupby_default_to_pandas(
                self, groupby_method_name, force_series=self._is_series, **kwargs)
        else:
            output_dataframe = self._qc.groupby_reduce(
                groupby_method_name, self, **kwargs)
        return output_dataframe

    def agg(self, arg=None, **kwargs):
        """
        This func is only used for demo purposes.
        """
        if arg == 'all':
            return self.all(**kwargs)
        if arg == 'any':
            return self.any(**kwargs)
        if arg == 'count':
            return self.count(**kwargs)
        if arg == 'max':
            return self.max(**kwargs)
        if arg == 'min':
            return self.min(**kwargs)
        if arg == 'prod':
            return self.prod(**kwargs)
        if arg == 'size':
            return self.size(**kwargs)
        if arg == 'sum':
            return self.sum(**kwargs)

        force_series = self.ndim == 1
        kwargs["func"] = arg
        result = self._qc.groupby_default_to_pandas(self, "agg", force_series, **kwargs)
        return result

    def all(self, skipna=True, **kwargs):
        kwargs['skipna'] = skipna
        if self._validate_dtypes():
            return self._qc.groupby_default_to_pandas(self, 'all', force_series=self._is_series,
                                                      **kwargs)
        return self.func_wrapper('all', **kwargs)

    def any(self, skipna=True, **kwargs):
        kwargs['skipna'] = skipna
        if self._validate_dtypes():
            return self._qc.groupby_default_to_pandas(self, 'any', force_series=self._is_series,
                                                      **kwargs)
        return self.func_wrapper('any', **kwargs)

    def backfill(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'backfill', force_series=self._is_series,
                                                  **kwargs)

    def bfill(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'bfill', force_series=self._is_series,
                                                  **kwargs)

    def boxplot(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'boxplot', force_series=self._is_series,
                                                  **kwargs)

    def corr(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'corr', force_series=self._is_series,
                                                  **kwargs)

    def corrwith(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'corrwith', force_series=self._is_series,
                                                  **kwargs)

    def cov(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'cov', force_series=self._is_series,
                                                  **kwargs)

    def count(self, **kwargs):
        if self._validate_dtypes():
            return self._qc.groupby_default_to_pandas(self, 'count', force_series=self._is_series,
                                                      **kwargs)
        return self.func_wrapper('count', **kwargs)

    def cumcount(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'cumcount', force_series=self._is_series,
                                                  **kwargs)

    def cummax(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'cummax', force_series=self._is_series,
                                                  **kwargs)

    def cummin(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'cummin', force_series=self._is_series,
                                                  **kwargs)

    def cumprod(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'cumprod', force_series=self._is_series,
                                                  **kwargs)

    def cumsum(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'cumsum', force_series=self._is_series,
                                                  **kwargs)

    def describe(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'describe', force_series=self._is_series,
                                                  **kwargs)

    def diff(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'diff', force_series=self._is_series,
                                                  **kwargs)

    def ffill(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'ffill', force_series=self._is_series,
                                                  **kwargs)

    def fillna(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'fillna', force_series=self._is_series,
                                                  **kwargs)

    def filter(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'filter', force_series=self._is_series,
                                                  **kwargs)

    def first(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'first', force_series=self._is_series,
                                                  **kwargs)

    def head(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'head', force_series=self._is_series,
                                                  **kwargs)

    def hist(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'hist', force_series=self._is_series,
                                                  **kwargs)

    def idxmax(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'idxmax', force_series=self._is_series,
                                                  **kwargs)

    def idxmin(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'idxmin', force_series=self._is_series,
                                                  **kwargs)

    def last(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'last', force_series=self._is_series,
                                                  **kwargs)

    def mad(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'mad', force_series=self._is_series,
                                                  **kwargs)

    def max(self, numeric_only=False, min_count=- 1, **kwargs):
        kwargs["numeric_only"] = numeric_only
        kwargs["min_count"] = min_count
        if numeric_only is not False or min_count != -1 or self._validate_dtypes():
            return self._qc.groupby_default_to_pandas(self, 'max', force_series=self._is_series,
                                                      **kwargs)
        return self.func_wrapper('max', **kwargs)

    def mean(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'mean', force_series=self._is_series,
                                                  **kwargs)

    def min(self, numeric_only=False, min_count=- 1, **kwargs):
        kwargs["numeric_only"] = numeric_only
        kwargs["min_count"] = min_count
        if numeric_only is not False or min_count != -1 or self._validate_dtypes():
            return self._qc.groupby_default_to_pandas(self, 'min', force_series=self._is_series,
                                                      **kwargs)
        return self.func_wrapper('min', **kwargs)

    def median(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'median', force_series=self._is_series,
                                                  **kwargs)

    def ngroup(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'ngroup', force_series=self._is_series,
                                                  **kwargs)

    def nth(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'nth', force_series=self._is_series,
                                                  **kwargs)

    def nunique(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'nunique', force_series=self._is_series,
                                                  **kwargs)

    def ohlc(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'ohlc', force_series=self._is_series,
                                                  **kwargs)

    def pad(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'pad', force_series=self._is_series,
                                                  **kwargs)

    def pct_change(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'pct_change', force_series=self._is_series,
                                                  **kwargs)

    def plot(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'plot', force_series=self._is_series,
                                                  **kwargs)

    def prod(self, numeric_only=no_default, min_count=0, **kwargs):
        kwargs['numeric_only'] = numeric_only
        kwargs['min_count'] = min_count
        if numeric_only is not no_default or min_count != 0 or self._validate_dtypes():
            return self._qc.groupby_default_to_pandas(self, 'prod', force_series=self._is_series,
                                                      **kwargs)
        return self.func_wrapper('prod', **kwargs)

    def quantile(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'quantile', force_series=self._is_series,
                                                  **kwargs)

    def rank(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'rank', force_series=self._is_series,
                                                  **kwargs)

    def resample(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'resample', force_series=self._is_series,
                                                  **kwargs)

    def sample(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'sample', force_series=self._is_series,
                                                  **kwargs)

    def sem(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'sem', force_series=self._is_series,
                                                  **kwargs)

    def shift(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'shift', force_series=self._is_series,
                                                  **kwargs)

    def size(self, **kwargs):
        result = self.func_wrapper('size', **kwargs)
        return result

    def skew(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'skew', force_series=self._is_series,
                                                  **kwargs)

    def sum(self, numeric_only=no_default, min_count=0, **kwargs):
        kwargs['numeric_only'] = numeric_only
        kwargs['min_count'] = min_count
        if numeric_only is not no_default or min_count != 0 or self._validate_dtypes():
            return self._qc.groupby_default_to_pandas(self, 'sum', force_series=self._is_series,
                                                      **kwargs)
        return self.func_wrapper('sum', **kwargs)

    def std(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'std', force_series=self._is_series,
                                                  **kwargs)

    def tail(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'tail', force_series=self._is_series,
                                                  **kwargs)

    def take(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'take', force_series=self._is_series,
                                                  **kwargs)

    def tshift(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'tshift', force_series=self._is_series,
                                                  **kwargs)

    def var(self, **kwargs):
        return self._qc.groupby_default_to_pandas(self, 'var', force_series=self._is_series,
                                                  **kwargs)

    def __getattribute__(self, item):
        """ bypass all the edge case which final result is same as original pandas result
            but the euqals() return false
            1. handling the edge case when contain categorical columns
        """
        attr = super().__getattribute__(item)
        if callable(attr):
            if self.default_to_pandas  and hasattr(pandas.core.groupby.GroupBy, item):
                def default_func(*args, **kwargs):
                    ret = self._qc.groupby_default_to_pandas(
                        self, item, force_series=self._is_series, *args, **kwargs)
                    return ret
                return default_func
        return attr


class SeriesGroupBy(DataFrameGroupBy):
    """
    A class that hold the groupby interfaces for Series
    """

    def __init__(self,
                 dataframe,
                 by=None,
                 axis=0,
                 level=None,
                 as_index=True,
                 sort=True,
                 group_keys=True,
                 squeeze=no_default,  # Deprecated argument since pandas version 1.1.0.
                 observed=False,
                 dropna=True,
                 by_names=None,
                 by_dataframe=None,
                 drop=False,
                 is_mapping=False,
                 default_to_pandas=False,
                 get_item_key=None,
                 **kwargs
                 ):
        if not as_index:
            raise RuntimeError(
                f"argument as_index {as_index} is valid only for DataFrame.")
        super(SeriesGroupBy, self).__init__(dataframe, by, axis, level, as_index, sort,
                                            group_keys, squeeze, observed, dropna, by_names,
                                            by_dataframe,
                                            drop=as_index,
                                            is_mapping=is_mapping,
                                            default_to_pandas=default_to_pandas,
                                            get_item_key=get_item_key,
                                            **kwargs)

    @property
    def ndim(self):
        return 1
