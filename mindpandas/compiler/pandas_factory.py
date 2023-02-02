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
"""Original pandas factory"""
import numpy as np
import pandas
from pandas.core.dtypes.missing import isna as missing_type_isna
from pandas.api.types import is_scalar
from pandas.core.dtypes.cast import (
    find_common_type,
    maybe_downcast_to_dtype,
)
import mindpandas as mpd


class SumCount:
    """sum of the values count non-NA cells over the requested axis."""

    def __init__(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.level = level
        self.numeric_only = numeric_only
        self.kwargs = kwargs

    def __call__(self, dataframe):
        result = pandas.DataFrame(
            {
                "sum": dataframe.sum(axis=self.axis, skipna=self.skipna, numeric_only=self.numeric_only),
                "count": dataframe.count(axis=self.axis, numeric_only=self.numeric_only),
            }
        )
        if '__unsqueeze_series__' in result.index:
            result = result.reset_index(drop=True)
        result = result if self.axis else result.T
        return result


class IsNA:
    """Detect missing values."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, dataframe):
        return pandas.DataFrame(dataframe.isna())


def dtypes():
    """Return the dtypes in the DataFrame."""
    return lambda dataframe: dataframe.dtypes


def dtypes_post():
    """Calculate and return the dtypes in the DataFrame."""
    def out_fn(dataframe):
        out_df = pandas.DataFrame()
        ls_len = None

        for col in dataframe:
            # Return a Series containing counts of unique values.
            ser = dataframe[col].value_counts()
            ser_ls = []
            for index in ser.index:
                temp_ser = pandas.Series(0).astype(index)
                ser_ls.append(temp_ser)
            ls_len = max(ls_len, len(ser_ls)) if ls_len else len(ser_ls)
            if len(ser_ls) != ls_len:
                ser_ls += [ser_ls[0]] * (ls_len - len(ser_ls))
            out_df[col] = pandas.Series(pandas.concat(ser_ls))

        return out_df.dtypes
    return lambda dataframe: dataframe[0] if isinstance(dataframe, list) else out_fn(dataframe)

    # def out_fn(df):
    #     cols = []
    #     out_df = pandas.DataFrame()
    #     ls_len = len(df.loc[[df.index[0]]])
    #     for i in df.index:
    #         if i in cols:
    #             break
    #         else:
    #             cols.append(i)
    #             ser = df.loc[[i]].value_counts()    # ser.index gives dtypes of df.loc[[i]]
    #             ls = []
    #             for t in ser.index:
    #                 temp_ser = pandas.Series(0).astype(t)
    #                 ls.append(temp_ser)
    #             if (ls_len is not len(ls)):
    #                 ls += [ls[0]] * (ls_len - len(ls))
    #             temp_df = pandas.DataFrame(np.array(ls), columns=[i])
    #             out_df = pandas.concat([out_df, temp_df], axis=1)
    #     return out_df.dtypes
    # return (lambda df: out_fn(df))


class IsIn:
    """Whether each element in the DataFrame is contained in values."""

    def __init__(self, **kwargs):
        self.values = kwargs["values"]

    def __call__(self, dataframe):
        return pandas.DataFrame(dataframe.isin(values=self.values))


class Dropna:
    """Remove missing values."""

    def __init__(self, axis, how, thresh, subset, inplace):
        self.axis = axis
        self.how = how
        self.thresh = thresh
        self.subset = subset
        self.inplace = inplace

    def __call__(self, dataframe):
        dataframe = dataframe.dropna(axis=self.axis, how=self.how, thresh=self.thresh,
                                     subset=self.subset, inplace=self.inplace)
        return dataframe


class ReduceMean:
    """return value of dataframe sum divided by dataframe count along given axis"""

    def __init__(self, axis=0, skipna=True, numeric_only=False, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.numeric_only = numeric_only
        self.kwargs = kwargs

    def __call__(self, dataframe):
        if self.axis:
            sum_val = dataframe["sum"]
            count_val = dataframe["count"]
        else:
            sum_val = dataframe.loc["sum"]
            count_val = dataframe.loc["count"]

        if not isinstance(sum_val, pandas.Series):
            sum_val = sum_val.sum(axis=self.axis, skipna=self.skipna)
            count_val = count_val.sum(axis=self.axis)

        if 0 in count_val.values or any(np.isnan(v) for v in count_val.values) or sum_val.empty or count_val.empty:
            raise TypeError("unsupported operand type(s)")

        return sum_val / count_val


class RollingMap:
    """Calculate the rolling operations based on given method_name, like sum, count, std..."""

    def __init__(self, method_name, **kwargs):
        self.method_name = method_name
        self.kwargs = kwargs

    def __call__(self, dataframe, window=None, min_periods=None, center=False,
                 win_type=None, on=None, axis=0, closed=None, method="single"):
        result = getattr(dataframe.rolling(window=window, min_periods=min_periods,
                                           center=center, win_type=win_type, on=on, axis=axis),
                         self.method_name)(**self.kwargs)
        return pandas.DataFrame(result)


class GroupbyMap:
    """Calculate the groupby operations based on given method_name, like sum, count, std..."""

    def __init__(self, method_name, **kwargs):
        self.method_name = method_name
        self.groupby_kwargs = kwargs.pop('groupby_kwargs', None)
        self.is_mapping = kwargs.pop('is_mapping', None)
        self.min_count = kwargs.pop('min_count', 0)
        self.kwargs = kwargs

    def __call__(self, dataframe, group_by):
        sort = self.groupby_kwargs['sort']   # need sort
        observed = self.groupby_kwargs['observed']
        dropna = self.groupby_kwargs['dropna']
        axis = self.groupby_kwargs['axis']
        level = self.groupby_kwargs['level']

        if observed:
            sort = False

        if level is not None:   # by is None
            result = getattr(dataframe.groupby(level=level,
                                               axis=axis,
                                               sort=sort,
                                               dropna=dropna),
                             self.method_name)(**self.kwargs)
            return result

        if axis == 0:
            if isinstance(group_by, pandas.DataFrame):
                dataframe = dataframe.drop(columns=group_by.columns, errors='ignore')  # drop group_by

                by_cols = list(group_by.columns)
                if len(by_cols) == 1:
                    # converting single column DataFrame by to Series
                    group_by = group_by[by_cols[0]]

                    # drop index if dataframe has index
                    dataframe = dataframe.reset_index(drop=True)
                    # reset by index to align with the dataframe index
                    group_by = group_by.reset_index(drop=True)
                else:
                    # drop index if dataframe has index
                    dataframe = dataframe.reset_index(drop=True)
                    # Add the by parts back to df
                    by_columns = group_by[[b for b in by_cols if b not in dataframe]]
                    # reset by index to align with the dataframe index
                    by_columns = by_columns.reset_index(drop=True)
                    dataframe = pandas.concat(
                        [dataframe] + [by_columns],
                        axis=1)
                    group_by = list(group_by.columns)   # change by to labels
            elif isinstance(group_by, (pandas.Series, mpd.Series)):
                dataframe = dataframe.drop(columns=group_by.columns, errors='ignore')  # drop group_by
                # Since Series can only used for mapping,
                # we convert it to a list so we can groupby by labels / index levels
                # NOTE: to_list() consume times, need to optimize later
                group_by = group_by.to_list()

            if self.is_mapping:
                # Case for mapping,
                # mapping need to be a list because list does not have label information
                group_by = group_by.to_list()
        else:
            group_by = [x for y in group_by.values.tolist() for x in y]
        result = getattr(dataframe.groupby(group_by, axis=axis, sort=sort, dropna=dropna),
                         self.method_name)(**self.kwargs)   # as_index handle in other place
        return pandas.DataFrame(result)


class GroupbyReduce:
    """Calculate the groupby operations based on given method_name, like sum, count, std...
       Return dataframe containing specific index names
    """

    def __init__(self, method_name, by_names, **kwargs):
        self.method_name = method_name
        self.by_names = by_names
        self.groupby_kwargs = kwargs.pop('groupby_kwargs', None)
        self.numeric_only = kwargs.pop('numeric_only', False)
        self.is_mapping = kwargs.pop('is_mapping', None)
        self.min_count = kwargs.pop('min_count', 0)
        self.kwargs = kwargs

    def __call__(self, dataframe):
        output_index_names = dataframe.index.names
        dropna = self.groupby_kwargs['dropna']
        sort = self.groupby_kwargs['sort']
        observed = self.groupby_kwargs['observed']
        # Saved index and used for rebuild the dataframe late
        axis = self.groupby_kwargs['axis']

        if axis == 0:
            group_by = dataframe.index
        else:
            group_by = dataframe.columns

        output_df = getattr(dataframe.groupby(
            group_by, axis=axis, sort=sort, dropna=dropna), self.method_name)(**self.kwargs)

        output = None
        if len(output_index_names) == 1:
            output = pandas.DataFrame(output_df)
            output.index.name = self.by_names[0] if isinstance(
                self.by_names, list) else self.by_names   # add index name
        else:
            # convert tuple-type index into multi-index
            if isinstance(output_df.index, pandas.Index) and isinstance(output_df.index[0], tuple):
                output_df_index = pandas.MultiIndex.from_tuples(
                    output_df.index.values, names=output_index_names)
                output = pandas.DataFrame(output_df, index=output_df_index)

                if observed:
                    # remove index with all zeros
                    output = output[(output.T != 0).any()]
            else:
                # fix the multi-index equals bugs here
                output = pandas.DataFrame(output_df, index=output_index_names)
        return output


class Abs:
    """Return a Series/DataFrame with absolute numeric value of each element."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.abs()


class Fillna:
    """Fill NA/NaN values using the specified method."""

    def __init__(self, **kwargs):
        self.value = kwargs.pop("value")
        self.kwargs = kwargs
        self.is_series = kwargs.pop("is_series", False)
        self.fill_with_primitive = False
        self.squeeze_self = kwargs.pop("squeeze_self", False)
        if is_scalar(self.value) or isinstance(self.value, str):
            self.fill_with_primitive = True

    def __call__(self, dataframe):
        if self.squeeze_self:
            return dataframe.squeeze(axis=1).fillna(value=self.value, **self.kwargs)
        if self.is_series and not self.fill_with_primitive:
            return dataframe.fillna(value=self.value.to_frame(name='__unsqueeze_series__'),
                                    **self.kwargs)
        return dataframe.fillna(value=self.value, **self.kwargs)


class Sum:
    """sum up the values over the requested axis"""

    def __init__(self, axis=None, skipna=True, numeric_only=None, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.numeric_only = numeric_only
        self.kwargs = kwargs

    def __call__(self, dataframe):
        result = pandas.DataFrame(
            {
                "sum": dataframe.sum(axis=self.axis, skipna=self.skipna,
                                     numeric_only=self.numeric_only)
            }
        )
        result = result if self.axis else result.T
        return result


class ReduceSum:
    """return sum values selected by min_count if given."""

    def __init__(self, axis=None, skipna=True, numeric_only=None, min_count=0, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.numeric_only = numeric_only
        self.min_count = min_count
        self.kwargs = kwargs

    def __call__(self, dataframe):
        if self.axis == 0:
            sum_val = dataframe.loc["sum"]
            count_val = dataframe.loc["count"]
            if self.skipna and sum_val.isna().any(axis=None):
                if isinstance(sum_val, pandas.Series):
                    sum_val = sum_val.to_frame().T
                    count_val = count_val.to_frame().T
                valid_mask = ~(sum_val.reset_index(drop=True).isna() & (count_val.reset_index(drop=True) > 0)).any()
                sum_val = sum_val.loc[:, valid_mask]
                count_val = count_val.loc[:, valid_mask]
        else:
            sum_val = dataframe["sum"]
            count_val = dataframe["count"]

        if not isinstance(sum_val, pandas.Series):
            sum_val = sum_val.sum(axis=self.axis, skipna=self.skipna)
        if self.min_count > 0:
            if not isinstance(count_val, pandas.Series):
                count_val = count_val.sum(axis=self.axis)
            sum_val = sum_val.where(count_val >= self.min_count)
        return sum_val


class Std:
    """Return sample standard deviation over requested axis."""

    def __init__(self, axis=0, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.level = level
        self.ddof = ddof
        self.numeric_only = numeric_only
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.std(dataframe, axis=self.axis, skipna=self.skipna, level=self.level, ddof=self.ddof,
                                    numeric_only=self.numeric_only, **self.kwargs)


class Count:
    """Return count non-NA cells for each column or row."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.count(**self.kwargs)


class Var:
    """Return unbiased variance over requested axis."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.var(**self.kwargs)


class Prod:
    """Return the product of the values over the requested axis."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.prod(**self.kwargs)


class Rename:
    """Alter axes labels."""

    def __init__(self, mapper=None, *, index=None, columns=None, axis=None,
                 copy=True, inplace=False, level=None, errors='ignore'):
        self.mapper = mapper
        self.index = index
        self.columns = columns
        self.axis = axis
        self.copy = copy
        self.inplace = inplace
        self.level = level
        self.errors = errors

    def __call__(self, dataframe):
        return pandas.DataFrame.rename(dataframe, mapper=self.mapper, index=self.index,
                                       columns=self.columns, axis=self.axis, copy=self.copy,
                                       inplace=self.inplace, level=self.level, errors=self.errors)


class Merge:
    """Merge DataFrame or named Series objects with a database-style join."""

    def __init__(self, other, **kwargs):
        self.kwargs = kwargs
        self.how = kwargs.get('how')
        self.other = other.backend_frame.to_pandas()

    def __call__(self, dataframe):
        if self.how in ['inner', 'left']:
            return pandas.merge(left=dataframe, right=self.other, **self.kwargs)
        if self.how == 'right':
            return pandas.merge(left=self.other, right=dataframe, **self.kwargs)
        return None


class CumSum:
    """Return cumulative sum over a DataFrame or Series axis."""

    def __init__(self, *args, axis=0, skipna=True, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.cumsum(dataframe, axis=self.axis, skipna=self.skipna,
                                       *self.args, **self.kwargs)


class All:
    """Return whether all elements are True."""

    def __init__(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        self.axis = axis
        self.bool_only = bool_only
        self.skipna = skipna
        self.level = level
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.all(dataframe,
                                    axis=self.axis,
                                    bool_only=self.bool_only,
                                    skipna=self.skipna,
                                    level=self.level,
                                    **self.kwargs)


class Any:
    """Return whether any element is True, potentially over an axis."""

    def __init__(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        self.axis = axis
        self.bool_only = bool_only
        self.skipna = skipna
        self.level = level
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.any(dataframe,
                                    axis=self.axis,
                                    bool_only=self.bool_only,
                                    skipna=self.skipna,
                                    level=self.level,
                                    **self.kwargs)


class Max:
    """Return the minimum of the values over the requested axis"""

    def __init__(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.level = level
        self.numeric_only = numeric_only
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.max(dataframe, axis=self.axis,
                                    skipna=self.skipna,
                                    level=self.level,
                                    numeric_only=self.numeric_only,
                                    **self.kwargs)


class Min:
    """Return the minimum of the values over the requested axis"""

    def __init__(self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.level = level
        self.numeric_only = numeric_only
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.min(dataframe, axis=self.axis,
                                    skipna=self.skipna,
                                    level=self.level,
                                    numeric_only=self.numeric_only,
                                    **self.kwargs)


class MathOp:
    """Do math operations along given axis"""

    def __init__(self, opr, axis='columns', level=None, fill_value=None):
        self.axis = axis
        self.level = level
        self.fill_value = fill_value
        if opr == 'add':
            self.func = pandas.DataFrame.add
        elif opr == 'sub':
            self.func = pandas.DataFrame.sub
        elif opr == 'mul':
            self.func = pandas.DataFrame.mul
        elif opr == 'div':
            self.func = pandas.DataFrame.div
        elif opr == 'truediv':
            self.func = pandas.DataFrame.truediv
        elif opr == 'floordiv':
            self.func = pandas.DataFrame.floordiv
        elif opr == 'mod':
            self.func = pandas.DataFrame.mod
        elif opr == 'pow':
            self.func = pandas.DataFrame.pow

    def __call__(self, dataframe, other):
        return self.func(dataframe, other, axis=self.axis,
                         level=self.level, fill_value=self.fill_value)


class SeriesComparison:
    """compare two series"""

    def __init__(self, func, level, fill_value, axis, is_scalar_local, *args, **kwargs):
        self.name = func
        if func == 'eq':
            self.func = pandas.Series.eq
        elif func == 'le':
            self.func = pandas.Series.le
        elif func == 'lt':
            self.func = pandas.Series.lt
        elif func == 'ge':
            self.func = pandas.Series.ge
        elif func == 'gt':
            self.func = pandas.Series.gt
        elif func == 'ne':
            self.func = pandas.Series.ne
        elif func == 'equals':
            self.func = pandas.Series.equals

        self.level = level
        self.fill_value = fill_value
        self.axis = axis
        self.is_scalar = is_scalar_local
        self.args = args
        self.kwargs = kwargs

    def __call__(self, series, other):
        if self.name == 'equals':
            # pandas.Series.equals returns a single bool,
            # need to wrap the single bool to assign index
            return pandas.Series(self.func(series, other))
        if self.is_scalar:
            if series.empty:
                series = pandas.Series(None)
                return self.func(series, other)
            return self.func(series.squeeze('columns'), other)
        if other.empty:
            other = pandas.Series(None)
            if series.empty:
                series = pandas.Series(None)
                return self.func(series, other)
            return self.func(series.squeeze('columns'), other)
        return self.func(series.squeeze('columns'), other.squeeze('columns'))


class DataFrameComparison:
    """compare two dataframes"""

    def __init__(self, func, level, axis, *args, **kwargs):
        if func == 'eq':
            self.func = pandas.DataFrame.eq
        elif func == 'le':
            self.func = pandas.DataFrame.le
        elif func == 'lt':
            self.func = pandas.DataFrame.lt
        elif func == 'ge':
            self.func = pandas.DataFrame.ge
        elif func == 'gt':
            self.func = pandas.DataFrame.gt
        elif func == 'ne':
            self.func = pandas.DataFrame.ne
        elif func == 'equals':
            self.func = pandas.DataFrame.equals

        self.level = level
        self.axis = axis
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dataframe, other):
        return self.func(dataframe, other)


class Combine:
    """Custom Combine call based on Pandas API's Combine."""

    def __init__(self, func, fill_value=None, overwrite=True, **kwargs):
        self.func = func
        self.fill_value = fill_value
        self.overwrite = overwrite
        self.kwargs = kwargs

    def __call__(self, dataframe, other, orig_df_cols, orig_other_cols):
        '''
        Removes redundant calls such as align and union from the Pandas version
        '''

        # df is a populated DataFrame while other may be empty
        if other.empty:
            return dataframe.copy()

        # Due to earlier align and repartitioning, dataframe will contain itself and other's columns
        new_columns = dataframe.columns
        need_fill = self.fill_value is not None

        result = {}
        for col in new_columns:
            ms_series = dataframe[col]
            ms_other_series = other[col]

            this_dtype = ms_series.dtype
            other_dtype = ms_other_series.dtype

            mask_this = missing_type_isna(ms_series)
            mask_other = missing_type_isna(ms_other_series)

            if not self.overwrite and mask_other.all():
                result[col] = dataframe[col].copy()
                continue

            if need_fill:
                ms_series = ms_series.copy()
                ms_other_series = ms_other_series.copy()
                ms_series[mask_this] = self.fill_value
                ms_other_series[mask_other] = self.fill_value

            if col not in orig_df_cols:
                updated_dtype = other_dtype
                try:
                    ms_series = ms_series.astype(updated_dtype, copy=False)
                except ValueError:
                    pass
            else:
                updated_dtype = find_common_type([this_dtype, other_dtype])
                ms_series = ms_series.astype(updated_dtype, copy=False)
                ms_other_series = ms_other_series.astype(updated_dtype, copy=False)

            arr = self.func(ms_series, ms_other_series)

            if isinstance(updated_dtype, np.dtype):
                arr = maybe_downcast_to_dtype(arr, updated_dtype)

            result[col] = arr

        return pandas.DataFrame.from_dict(result)


class Explode:
    """
    This func is only used for demo purposes.

    Custom explode call based on Pandas API's explode.
    """
    def __init__(self, column, ignore_index=False):
        self.column = column
        self.ignore_index = ignore_index

    def __call__(self, dataframe):
        return pandas.DataFrame.explode(dataframe,
                                        column=self.column,
                                        ignore_index=self.ignore_index)


class CumMin:
    """Return cumulative minimum over a DataFrame or Series axis."""

    def __init__(self, *args, axis=0, skipna=True, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.cummin(dataframe, axis=self.axis, skipna=self.skipna,
                                       *self.args, **self.kwargs)


class CumMax:
    """Return cumulative maximum over a DataFrame or Series axis."""

    def __init__(self, *args, axis=0, skipna=True, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.cummax(dataframe, axis=self.axis, skipna=self.skipna,
                                       *self.args, **self.kwargs)


class CumProd:
    """Return cumulative product over a DataFrame or Series axis."""

    def __init__(self, *args, axis=0, skipna=True, **kwargs):
        self.axis = axis
        self.skipna = skipna
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame.cumprod(dataframe, axis=self.axis, skipna=self.skipna,
                                        *self.args, **self.kwargs)


class Apply:
    """Apply a function along an axis of the DataFrame."""

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.convert_dtype = kwargs.pop('convert_dtype', None)
        self.apply_to_series = bool(self.convert_dtype is not None)

    def __call__(self, dataframe):
        if self.apply_to_series:
            return dataframe.squeeze(axis=1).apply(self.func, convert_dtype=self.convert_dtype,
                                                   **self.kwargs)
        return dataframe.apply(self.func, **self.kwargs)


class Insert:
    """Insert column into DataFrame at specified location."""

    def __init__(self, **kwargs):
        self.loc = kwargs.get('loc')
        self.column = kwargs.get('column')
        self.value = kwargs.get('value')
        self.allow_duplicates = kwargs.get('allow_duplicates')

    def __call__(self, dataframe):
        return dataframe.insert(loc=self.loc,
                                column=self.column,
                                value=self.value,
                                allow_duplicates=self.allow_duplicates)


class Rank:
    """Compute numerical data ranks along axis."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.DataFrame(dataframe.rank(**self.kwargs))


class ToDateTime:
    """Convert argument to datetime."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return pandas.to_datetime(dataframe.squeeze(axis=1), **self.kwargs)


class Reindex:
    """Conform Series/DataFrame to new index."""

    def __init__(self, **kwargs):
        self.labels = kwargs.pop('labels')
        self.axis = kwargs.pop('axis')
        self.kwargs = kwargs

    def __call__(self, dataframe):
        dataframe = pandas.DataFrame(dataframe.reindex(
            labels=self.labels, axis=self.axis, **self.kwargs))
        return dataframe


class ApplyMap:
    """Apply a function to a Dataframe elementwise."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.applymap(**self.kwargs)


class Transpose:
    """Transpose index and columns."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.transpose(**self.kwargs)


class AsType:
    """Cast a pandas object to a specified dtype dtype."""

    def __init__(self, **kwargs):
        self.dtype = kwargs.get('dtype')
        self.copy = kwargs.get('copy')
        self.errors = kwargs.get('errors')

    def __call__(self, dataframe):
        if isinstance(self.dtype, dict):
            return dataframe.astype({k: v for k, v in self.dtype.items() if k in dataframe},
                                    self.copy, self.errors)
        return dataframe.astype(self.dtype, self.copy, self.errors)


class SetItemPartition:
    """set values to given internal indices"""

    def __init__(self, axis, value, **kwargs):
        self.axis = axis
        self.value = value
        self.kwargs = kwargs

    def __call__(self, dataframe, internal_indices=None):
        dataframe = dataframe.copy()
        if len(internal_indices) == 1:
            if self.axis == 0:
                dataframe[dataframe.columns[internal_indices[0]]] = self.value
            else:
                dataframe.iloc[internal_indices[0]] = self.value
        else:
            if self.axis == 0:
                dataframe[dataframe.columns[internal_indices]] = self.value
            else:
                dataframe.iloc[internal_indices] = self.value
        return dataframe


class Duplicated:
    """Return boolean Series denoting duplicate rows."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.duplicated(**self.kwargs)


class Where:
    """Replace values where the condition is False."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe, cond, other):
        return dataframe.where(cond=cond, other=other, **self.kwargs)


class Mask:
    """Replace values where the condition is True."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe, cond, other):
        return dataframe.mask(cond=cond, other=other, **self.kwargs)


class Median:
    """Return the median of the values over the requested axis."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.median(**self.kwargs)


class Replace:
    """Replace values in the dataframe"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, data):
        return data.replace(**self.kwargs)


def to_list():
    """Convert dataframe values to list. """
    return lambda dataframe: dataframe.values.tolist()


class SetItem:
    """set values to given columns"""

    def __init__(self, columns, val=None):
        self.val = val
        self.columns = columns
        if isinstance(columns, list):
            self.columns = columns[0]
            print("Warning Setitem expects only one column, not list")

    def __call__(self, dataframe, other=None, const=None):
        dataframe = dataframe.copy()
        if other is None:
            other = self.val
        if isinstance(other, pandas.DataFrame):
            other = other.squeeze(1)
        if self.columns in dataframe.columns:
            dataframe[self.columns] = other
        return dataframe


class SetItemElements:
    """set particular element in the dataframe"""

    def __init__(self, method_name, **kwargs):
        self.method_name = method_name
        self.kwargs = kwargs

    def __call__(self, dataframe, *args):
        dataframe = pandas.concat([dataframe])
        getattr(dataframe, self.method_name)[args[0], args[1]] = args[2]
        return dataframe


class GetItem:
    """get columns based on given key"""

    def __init__(self, **kwargs):
        self.is_scalar = kwargs.get('is_scalar', False)
        self.kwargs = kwargs

    def __call__(self, dataframe, key, other=None):
        if self.is_scalar:
            key = key.squeeze(1)
        dataframe = dataframe[key]
        return dataframe


class Invert:
    """ implementation of the "binary not" operator ~"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, dataframe):
        return dataframe.__invert__()


class Concat:
    """Concatenate pandas objects along a particular axis."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, data, axis=0):
        return pandas.concat(data, axis)


class SortIndex:
    """Sort the index of the DataFrame."""

    def __init__(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last',
                 sort_remaining=True, ignore_index=False, key=None):
        self.axis = axis
        self.level = level
        self.ascending = ascending
        self.inplace = inplace
        self.kind = kind
        self.na_position = na_position
        self.sort_remaining = sort_remaining
        self.ignore_index = ignore_index
        self.key = key

    def __call__(self, dataframe):
        dataframe = pandas.DataFrame.sort_index(dataframe,
                                                axis=self.axis,
                                                level=self.level,
                                                ascending=self.ascending,
                                                inplace=self.inplace,
                                                kind=self.kind,
                                                na_position=self.na_position,
                                                sort_remaining=self.sort_remaining,
                                                ignore_index=self.ignore_index,
                                                key=self.key)
        return dataframe


class ResetIndex:
    """Reset the index of the DataFrame."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe, **kwargs):
        dataframe = dataframe.reset_index(**kwargs)
        return dataframe


class Drop:
    """Drop specified labels from rows or columns."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, dataframe):
        return dataframe.drop(errors='ignore', **self.kwargs)


class SetSeriesName:
    """Reset the name of the Series."""

    def __init__(self, value):
        self.value = value

    def __call__(self, dataframe):
        # Handling dataframe here because each partition is a dataframe type in the backend
        dataframe.columns = [self.value]
        return dataframe


class MaskILoc:
    """select rows/columns based on given index"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, data, row_indices, column_indices, is_series=False):
        '''Apply mask on data using specified indices.'''
        output = data.iloc[row_indices, column_indices]
        output_shape = output.shape
        if is_series and len(output_shape) > 1 and output_shape[1] == 1:
            output = output.squeeze("columns")
        return output


class MemoryUsage:
    """Return the memory usage of each column in bytes"""

    def __init__(self, index=True, deep=False):
        self.index = index
        self.deep = deep

    def __call__(self, dataframe):
        return pandas.DataFrame.memory_usage(dataframe, index=self.index, deep=self.deep)
