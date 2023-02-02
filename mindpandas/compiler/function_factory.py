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
""" Mindpandas Function Factory Class"""
import warnings

import mindpandas.internal_config as i_config


class FunctionFactory:
    """ Mindpandas Function Factory Class"""
    ff_: None
    if i_config.functions == "pandas":
        import mindpandas.compiler.pandas_factory as pf
        ff_ = pf
    else:
        warnings.warn("function factory type not implemented")

    @classmethod
    def sum_count(cls, axis=0, skipna=True, level=None, numeric_only=False, **kwargs):
        return cls.ff_.SumCount(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)

    @classmethod
    def reduce_mean(cls, axis=0, skipna=True, numeric_only=False, **kwargs):
        return cls.ff_.ReduceMean(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @classmethod
    def groupby_map(cls, method_name, **kwargs):
        return cls.ff_.GroupbyMap(method_name, **kwargs)

    @classmethod
    def rolling_map(cls, method_name, **kwargs):
        return cls.ff_.RollingMap(method_name, **kwargs)

    @classmethod
    def groupby_reduce(cls, method_name, by_names, **kwargs):
        return cls.ff_.GroupbyReduce(method_name, by_names, **kwargs)

    @classmethod
    def abs(cls, **kwargs):
        return cls.ff_.Abs(**kwargs)

    @classmethod
    def fill_na(cls, **kwargs):
        return cls.ff_.Fillna(**kwargs)

    @classmethod
    def isna(cls, **kwargs):
        return cls.ff_.IsNA(**kwargs)

    @classmethod
    def dtypes(cls, **kwargs):
        return cls.ff_.dtypes(**kwargs)

    @classmethod
    def dtypes_post(cls, **kwargs):
        return cls.ff_.dtypes_post(**kwargs)

    @classmethod
    def isin(cls, **kwargs):
        return cls.ff_.IsIn(**kwargs)

    @classmethod
    def dropna(cls, axis, how, thresh, subset, inplace):
        return cls.ff_.Dropna(axis, how, thresh, subset, inplace)

    @classmethod
    def sum(cls, axis=None, skipna=True, numeric_only=None, **kwargs):
        return cls.ff_.Sum(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @classmethod
    def reduce_sum(cls, axis=None, skipna=True, numeric_only=None, min_count=0, **kwargs):
        return cls.ff_.ReduceSum(axis=axis,
                                 skipna=skipna,
                                 numeric_only=numeric_only,
                                 min_count=min_count,
                                 **kwargs)

    @classmethod
    def std(cls, axis=None, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
        return cls.ff_.Std(axis=axis,
                           skipna=skipna,
                           level=level,
                           ddof=ddof,
                           numeric_only=numeric_only,
                           **kwargs)

    @classmethod
    def count(cls, **kwargs):
        return cls.ff_.Count(**kwargs)

    @classmethod
    def var(cls, **kwargs):
        return cls.ff_.Var(**kwargs)

    @classmethod
    def prod(cls, **kwargs):
        return cls.ff_.Prod(**kwargs)

    @classmethod
    def rename(cls,
               mapper=None,
               *,
               index=None,
               columns=None,
               axis=None,
               copy=True,
               inplace=False,
               level=None,
               errors='ignore'):
        """Alter axes labels"""
        return cls.ff_.Rename(mapper=mapper,
                              index=index,
                              columns=columns,
                              axis=axis,
                              copy=copy,
                              inplace=inplace,
                              level=level,
                              errors=errors)

    @classmethod
    def merge(cls, other, **kwargs):
        return cls.ff_.Merge(other, **kwargs)

    @classmethod
    def combine(cls, func, fill_value=None, overwrite=True, **kwargs):
        return cls.ff_.Combine(func=func,
                               fill_value=fill_value,
                               overwrite=overwrite,
                               **kwargs)

    @classmethod
    def explode(cls, column, ignore_index=False):
        return cls.ff_.Explode(column=column, ignore_index=ignore_index)

    @classmethod
    def series_comparison(cls, func, level, fill_value, axis, is_scalar, *args, **kwargs):
        return cls.ff_.SeriesComparison(func=func,
                                        level=level,
                                        fill_value=fill_value,
                                        axis=axis,
                                        is_scalar_local=is_scalar,
                                        *args,
                                        **kwargs)

    @classmethod
    def df_comparison(cls, func, level, axis, *args, **kwargs):
        return cls.ff_.DataFrameComparison(func=func, level=level, axis=axis, *args, **kwargs)

    @classmethod
    def cumsum(cls, axis=0, skipna=True, **kwargs):
        return cls.ff_.CumSum(axis=axis, skipna=skipna, **kwargs)

    @classmethod
    def cummin(cls, axis=0, skipna=True, **kwargs):
        return cls.ff_.CumMin(axis=axis, skipna=skipna, **kwargs)

    @classmethod
    def cummax(cls, axis=0, skipna=True, **kwargs):
        return cls.ff_.CumMax(axis=axis, skipna=skipna, **kwargs)

    @classmethod
    def cumprod(cls, axis=0, skipna=True, **kwargs):
        return cls.ff_.CumProd(axis=axis, skipna=skipna, **kwargs)

    @classmethod
    def apply(cls, func, **kwargs):
        return cls.ff_.Apply(func, **kwargs)

    @classmethod
    def all(cls, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        return cls.ff_.All(axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

    @classmethod
    def any(cls, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        return cls.ff_.Any(axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

    @classmethod
    def max(cls, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        return cls.ff_.Max(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)

    @classmethod
    def min(cls, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        return cls.ff_.Min(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs)

    @classmethod
    def math_op(cls, opr, axis='columns', level=None, fill_value=None):
        return cls.ff_.MathOp(opr=opr, axis=axis, level=level, fill_value=fill_value)

    @classmethod
    def insert(cls, **kwargs):
        return cls.ff_.Insert(**kwargs)

    @classmethod
    def rank(cls, **kwargs):
        return cls.ff_.Rank(**kwargs)

    @classmethod
    def to_datetime(cls, **kwargs):
        return cls.ff_.ToDateTime(**kwargs)

    @classmethod
    def reindex(cls, **kwargs):
        return cls.ff_.Reindex(**kwargs)

    @classmethod
    def applymap(cls, **kwargs):
        return cls.ff_.ApplyMap(**kwargs)

    @classmethod
    def transpose(cls, **kwargs):
        return cls.ff_.Transpose(**kwargs)

    @classmethod
    def astype(cls, **kwargs):
        return cls.ff_.AsType(**kwargs)

    @classmethod
    def setitem_part_func(cls, axis, value, **kwargs):
        return cls.ff_.SetItemPartition(axis=axis, value=value, **kwargs)

    @classmethod
    def duplicated(cls, **kwargs):
        return cls.ff_.Duplicated(**kwargs)

    @classmethod
    def where(cls, **kwargs):
        return cls.ff_.Where(**kwargs)

    @classmethod
    def mask(cls, **kwargs):
        return cls.ff_.Mask(**kwargs)

    @classmethod
    def median(cls, **kwargs):
        return cls.ff_.Median(**kwargs)

    @classmethod
    def replace(cls, **kwargs):
        return cls.ff_.Replace(**kwargs)

    @classmethod
    def to_list(cls):
        return cls.ff_.to_list()

    @classmethod
    def set_item(cls, columns, value=None):
        return cls.ff_.SetItem(columns, value)

    @classmethod
    def set_item_elements(cls, method_name, **kwargs):
        return cls.ff_.SetItemElements(method_name, **kwargs)

    @classmethod
    def get_item(cls, **kwargs):
        return cls.ff_.GetItem(**kwargs)

    @classmethod
    def invert(cls):
        return cls.ff_.Invert()

    @classmethod
    def concat(cls):
        return cls.ff_.Concat()

    @classmethod
    def sort_index(cls, axis, level, ascending, inplace, kind, na_position, sort_remaining, ignore_index, key):
        return cls.ff_.SortIndex(axis=axis,
                                 level=level,
                                 ascending=ascending,
                                 inplace=inplace,
                                 kind=kind,
                                 na_position=na_position,
                                 sort_remaining=sort_remaining,
                                 ignore_index=ignore_index,
                                 key=key)

    @classmethod
    def reset_index(cls):
        return cls.ff_.ResetIndex()

    @classmethod
    def drop(cls, index, columns):
        return cls.ff_.Drop(index=index, columns=columns)

    @classmethod
    def set_series_name(cls, value):
        return cls.ff_.SetSeriesName(value=value)

    @classmethod
    def mask_iloc(cls):
        return cls.ff_.MaskILoc()

    @classmethod
    def memory_usage(cls, index=True, deep=False):
        return cls.ff_.MemoryUsage(index=index, deep=deep)
