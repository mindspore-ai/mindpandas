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
Module contains ``QueryCompiler`` class which is responsible for compiling DataFrame algebra
queries for the MindPandas ``EagerFrame``.
"""

import hashlib
from typing import Dict, Iterable, Iterator, cast
import warnings

import numpy as np
import pandas
from pandas._libs.lib import no_default, maybe_convert_objects
from pandas.core import algorithms
from pandas.core.common import is_bool_indexer, apply_if_callable
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import is_list_like, is_scalar, is_integer_dtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.indexes.api import MultiIndex, ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.core.reshape.pivot import _add_margins
from pandas.core.reshape.util import cartesian_product
from pandas.io.common import get_handle

import mindpandas as mpd
import mindpandas.internal_config as i_config
from mindpandas.backend.base_general import BaseGeneral
from mindpandas.backend.base_io import BaseIO
from mindpandas.compiler.function_factory import FunctionFactory as ff
from mindpandas.util import is_boolean_array


class QueryCompiler:
    """
    QueryCompiler is responsible for translating common query compiler API into the DataFrame algebra queries
    which will be executed by MindPandas ``EagerFrame``
    """

    @classmethod
    def read_csv(cls, filepath, **kwargs):
        """Compiling read_csv using backend IO read_csv operator"""
        if isinstance(filepath, (int, bool)):
            raise ValueError(f"Invalid file path or buffer object type: {type(filepath)}")
        frame = BaseIO.read_csv(filepath, **kwargs)
        return mpd.DataFrame(data=frame)

    @classmethod
    def read_feather(cls, filepath, **kwargs):
        """Compiling read_feather using backend IO read_feather operator"""
        frame = BaseIO.read_feather(filepath, **kwargs)
        return mpd.DataFrame(data=frame)

    @classmethod
    def groupby(
            cls, input_dataframe,
            by=None,
            axis=0,
            level=None,
            as_index=True,
            sort=True,
            group_keys=True,
            squeeze: bool = no_default,
            observed=False,
            dropna=True
    ):
        """
        Compiling groupby query into DataFrameGroupby or SeriesDataFrameGroupby
        """

        # Validate parameters
        by_dataframe = None
        by_names = None
        is_mapping = False
        drop = as_index
        default_to_pandas = False
        mask_func = ff.mask_iloc()

        # NOTE:
        # by could be
        #     1) index levels: df.index.names
        #     2) column labels: df.columns

        #     Example:
        #     This a multi-index dataframe:
        #                         Max Speed  Max Speed2 String
        #     Animal    Type
        #     Wolverine Wild        1000.0         2.0      a
        #     Falcon    Captive      390.0        30.0      b
        #             Captive      350.0        30.0      c
        #     Parrot    Captive       30.0        10.0      d
        #             Wild          20.0        50.0      e

        #     1) index levels are: 'Animal' and 'Type'
        #     2) column labels are: 'Max Speed'  'Max Speed2' 'String'

        def is_label(by):
            return isinstance(by, (str, int))

        input_is_series = isinstance(input_dataframe, mpd.Series)
        if not input_is_series:
            column_labels = input_dataframe.columns.tolist()
        else:
            column_labels = [input_dataframe.name] if input_dataframe.name is not None else []

        # NOTE:
        # by: list of labels to groupby (used to drop to speed up)
        # by_dataframe (do not use series): backend_frame(eager_frame) contains:
        #     1) groupby dataframe
        #     2) a series of labels so we can pass as the by parameter when calling the original pandas
        #         (implement in pandas_factory/GroupbyMap)
        # by_names: original order of columns:
        #             list of labels: used to restore the order after all the manipulation
        #             (if not want to reorder, then set to None to speed up)

        if isinstance(by, pandas.Grouper):
            drop = True

        elif by is None and level is not None:
            # Default_to_pandas functions: we only pass level
            # func_wrapper functions: we will need to use by for the dataframe manipulations
            # NOTE: level is index.name, excluding column labels

            if input_is_series:  # Input is Series with no index level
                by_dataframe = cls.create_backend_frame(input_dataframe.index, index=None, columns=None,
                                                        dtype=None, copy=None)
                is_mapping = True
            else:
                # Setting all level cases default to pandas observes a better performance
                default_to_pandas = True
        elif is_label(by):
            # Groupby a column label or a index level
            if (axis == 0 and by in column_labels) or input_is_series:  # groupby a column labels
                # Groupby a column label
                by_dataframe = input_dataframe.backend_frame.get_columns(by, func=mask_func)   # only send df
                by_names = by
            elif (axis == 0) or input_is_series:    # by in index_names
                # Groupby a level label
                # Setting this cases default to pandas observes a better performance
                default_to_pandas = True
            elif axis == 1:
                column_level = [input_dataframe.columns.name] if input_dataframe.columns.name is not None else []
                if by in column_level:
                    default_to_pandas = True
                else:
                    mask_func = ff.mask_iloc()
                    by_dataframe = input_dataframe.backend_frame.get_rows(by, func=mask_func)   # only send df
                    by_names = by
            else:
                raise TypeError(f"'{by.__class__.__name__}' object is not callable")
        elif isinstance(by, list):
            # Groupby a list
            if not by:
                raise ValueError("No group keys passed!")

            index_names = input_dataframe.backend_frame.index_names_from_partitions()
            index_names = index_names if isinstance(index_names, list) else [index_names]
            all_labels = (column_labels if column_labels else []) + (index_names if index_names else [])

            if all(is_label(b) or b is np.nan or b is None for b in by):
                if (axis == 0 and all(is_label(b) and b in all_labels for b in by)) and (not input_is_series):
                    # Groupby a list of labels
                    if axis == 0 and all(b in column_labels for b in by):   # groupby a column labels
                        by_dataframe = input_dataframe.backend_frame.get_columns(by, func=mask_func)
                        by_names = by
                    elif axis == 0:  # groupby an index levels
                        # NOTE: by_dataframe can be an eager frame contains a series of labels so we can pass as the
                        #       by parameter when calling the original pandas
                        level = by
                    else:
                        default_to_pandas = True
                # elif all(is_label(b) or b is np.nan or b is None for b in by):
                else:
                    # Groupby a list of (str / int)

                    # Example:
                    # df.groupby([1, 2, 3, 4]).sum()
                    frame = input_dataframe.backend_frame
                    axes_len = frame.num_rows if axis == 0 else frame.num_cols
                    mismatch = axes_len != len(by)

                    if axis == 0 and not mismatch:
                        by_dataframe = cls.create_backend_frame(by, index=None, columns=None, dtype=None, copy=None)
                        is_mapping = True   # set is_mapping to True will convert by_dataframe back to list in pandas_factory
                    elif axis == 1 and not mismatch:
                        by_dataframe = cls.create_backend_frame([by], index=None, columns=None, dtype=None, copy=None)
                        is_mapping = False
                    else:
                        raise KeyError(by[0])

            elif all((isinstance(b, (dict, mpd.Series, pandas.Series)) for b in by)):
                # Groupby a list of (dict / Series) for mapping (mapping colmnms by orders), we need to transpose the
                # mapping list
                # NOTE: Need to speed up
                def transpose_list(by):
                    # Transpose a 2d list without changing elements' type
                    # lst = np.array([[y for y in x.values()] for x in by]).T cannot be used since it will auto
                    # convert elements' type

                    lst = [[] for x in range(len(by[0]))]
                    for i in range(len(by)):
                        for j in range(len(by[i])):
                            lst[j].append(by[i][j])
                    return lst

                if isinstance(by[0], dict):  # Case: List of Dict
                    # Groupby a list of dict for mapping
                    # Dict mapping does not have label name, so dict only pick columns by order

                    # Example:
                    # The DataFrame is:
                    #   Animal  Color  Height  Max Speed
                    # a  Falcon  black       3      380.0
                    # b  Falcon  white       2      370.0
                    # c  Parrot  black       2       24.0
                    # d  Parrot  white       3       26.0

                    # df.groupby([{'a': 'Falcon', 'b': 'Falcon', 'c': 'Parrot', 'd': 'Parrot'},
                    #             {'a': 'black', 'b': 'white', 'c': 'white'}]).sum()
                    #               Height  Max Speed
                    # Falcon black       3      380.0
                    #        white       2      370.0
                    # Parrot white       2       24.0

                    is_multi_dict = len(by) != 1
                    values, indices = [], []
                    for i, v in zip(input_dataframe.index, input_dataframe.values):
                        mapped_index = []
                        for d in by:
                            mapped_index.append(d.get(i))
                        if all(m is not None for m in mapped_index):
                            indices.append(mapped_index)
                            values.append(v)

                    if is_multi_dict:
                        # Note: multiindex case, default to pandas
                        default_to_pandas = True
                    else:
                        indices = [i[0] for i in indices]
                        index = pandas.Index(indices)
                        input_dataframe = mpd.DataFrame(values, index=index, columns=input_dataframe.columns)
                        by_dataframe = cls.create_backend_frame(index, index=None, columns=None, dtype=None, copy=None)
                        is_mapping = True
                elif isinstance(by[0], (mpd.Series, pandas.Series)):
                    # Groupby a list of Series for mapping

                    # Example:
                    # df.groupby([df['Animal'], df['Color']]).sum()

                    lst = transpose_list(by)
                    col = [x.backend_frame.to_pandas(force_series=True).name for x in by]

                    by_dataframe = cls.create_backend_frame(lst, index=None, columns=col, dtype=None, copy=None)
            else:
                raise TypeError(f"'{by[0].__class__.__name__}' object is not callable")

        elif isinstance(by, (mpd.Series, pandas.Series, Dict, np.ndarray)):
            # Groupby Series, nparray and Dict,
            # Those data type only support mapping

            if (len(by) == len(input_dataframe)
                    or (not input_is_series and len(by) == len(input_dataframe.columns))
                    or isinstance(by, np.ndarray)):
                #     Mapping each elements in by to each index in input_dataframe
                #     Injective mapping (means have the same length as DataFrame) series / Dict / np.ndarray

                #     Injective mapping series Example:
                #     The DataFrame is:
                #         Animal   Color   Height  Speed
                #     a   Falcon  black     3      380.
                #     b   Falcon  white     2      370.
                #     c   Parrot  black     2      24.
                #     d   Parrot  white     3      26.

                #     The result of df.groupby(Series([1, 2, 3, 3])).sum() is:
                #         Height  Max Speed
                #     1       3      380.0
                #     2       2      370.0
                #     3       5       50.0

                #     Injective mapping Dict Example:
                #     Same DataFrame above

                #     The result of df.groupby({'a': 'Falcon', 'b': 'Falcon', 'c': 'Parrot', 'd': 'Parrot'}).count() is:
                #     NOTE: abcd is the index of df, the index must be consistent
                #             Animal  Color  Height  Max Speed
                #     Falcon       2      2       2          2
                #     Parrot       2      2       2          2

                if isinstance(by, (Dict, mpd.Series, pandas.Series)):
                    if axis == 0:
                        if isinstance(by, Dict):
                            temp_by = by.values()
                            temp_columns = None
                        if isinstance(by, mpd.Series):
                            # NOTE: by is a list of label which used to drop the key column to speed up
                            temp_by = by
                            temp_columns = by.backend_frame.columns
                        if isinstance(by, pandas.Series):
                            # NOTE: by is a list of label which used to drop the key column to speed up
                            temp_by = by
                            temp_columns = by.name
                    else:
                        temp_by = [[b for b in by]]
                        temp_columns = None

                    by_dataframe = cls.create_backend_frame(temp_by, index=None, columns=temp_columns,
                                                            dtype=None, copy=None)
                else:  # np.ndarray
                    # Injective mapping np.ndarray

                    # Injective mapping np.ndarray Example:
                    # The DataFrame is:
                    #     Animal   Color   Height  Speed
                    # a   Falcon  black     3      380.
                    # b   Falcon  white     2      370.
                    # c   Parrot  black     2      24.
                    # d   Parrot  white     3      26.

                    # The result of df.groupby(np.array([1, 2, 3, 3])).sum() is:
                    #     Height  Max Speed
                    # 1       3      380.0
                    # 2       2      370.0
                    # 3       5       50.0
                    if axis == 0 and len(by) == len(input_dataframe.backend_frame.index):
                        by_dataframe = cls.create_backend_frame(by, index=None, columns=None, dtype=None, copy=None)
                        is_mapping = True   # set is_mapping to True will convert by_dataframe back to list in pandas_factory
                    elif axis == 1 and len(by) == len(input_dataframe.backend_frame.columns):
                        by_dataframe = cls.create_backend_frame([by], index=None, columns=None, dtype=None, copy=None)
                        is_mapping = False
                    else:
                        raise KeyError(by[0])
            else:   # mapping each key of dict/ser to the value of dict/ser; e.g.{'falcon': 'falcon_sec', 'lion': 'lion_sec'}
                # Mapping each key of (dict/ser) in each index to the value of (dict/ser)

                # Example:
                # The DataFrame is:
                #             class           order  max_speed
                # falcon     bird   Falconiformes      389.0
                # parrot     bird  Psittaciformes       24.0
                # lion     mammal       Carnivora       80.2
                # monkey   mammal        Primates        NaN
                # leopard  mammal       Carnivora       58.0

                # The result of df.groupby({'falcon': 'falcon_sec', 'lion': 'lion_sec'})).count() is:
                #                 class  order  max_speed
                # falcon_sec      1      1          1
                # lion_sec        1      1          1
                temp_by = by
                if not isinstance(by, Dict):  # Convert to Dict if by is a Series
                    temp_by = by.to_dict()

                if not input_is_series and axis == 0:
                    # Edge case: groupby by index without index.name
                    # Note: I was trying to use
                    # by_dataframe = cls.create_backend_frame(dict_values, index=None, columns=None, dtype=None,
                    # copy=None)
                    # but it shows 'IndexError: index 1 is out of bounds for axis 0 with size 1' in
                    # multiprocess_operators groupby_map()
                    # Thus, reset_index() put the index to column and then drop it later
                    dict_values = list(temp_by.values())
                    renamed_input_dataframe = input_dataframe.rename(index=temp_by)
                    filtered_input_dataframe = renamed_input_dataframe.loc[dict_values]  # loc mess up the partitions sizes
                    filtered_input_dataframe = filtered_input_dataframe.reset_index()
                    index_reseted = filtered_input_dataframe.columns[0]
                    by_dataframe = filtered_input_dataframe.backend_frame.get_columns(index_reseted, func=mask_func)
                    by_names = index_reseted
                    is_mapping = True

                elif not input_is_series and axis == 1:
                    new_columns = []
                    cols_index = []
                    for i, label in enumerate(column_labels):
                        new_col = by.get(label)
                        if new_col is not None:
                            new_columns.append(new_col)
                            cols_index.append(i)
                    output_frame = input_dataframe.backend_frame
                    rows_index = list(range(output_frame.num_rows))
                    mask_func = ff.mask_iloc()
                    output_frame = output_frame.view(rows_index_numeric=rows_index, columns_index_numeric=cols_index, func=mask_func)  # Drop unused columns
                    output_frame = cls.create_backend_frame(output_frame.to_pandas(), index=None, columns=None,
                                                            dtype=None, copy=None)

                    input_dataframe = mpd.DataFrame(output_frame)
                    by_dataframe = cls.create_backend_frame([new_columns], index=None, columns=None, dtype=None,
                                                            copy=None)  # df with one row
                    is_mapping = False

                else:
                    # Example:
                    # This is our series:
                    #     Falcon       390.0
                    #     Wolverine    350.0
                    #     Parrot        30.0
                    #     Cheetah       20.0
                    #     Name: Max Speed, dtype: float64

                    # The result of ser.groupby({'Falcon': 'Falcon1', 'Parrot': 'Parrot1'} is:
                    #     Falcon1    390.0
                    #     Parrot1      30.0
                    #     Name: Max Speed, dtype: float64

                    values, indices = [], []
                    for i, v in zip(input_dataframe.index, input_dataframe.values):
                        mapped_index = by.get(i)
                        if mapped_index is not None:
                            indices.append(mapped_index)
                            values.append(v)
                    index = pandas.Index(indices, name=input_dataframe.index.name)
                    input_dataframe = mpd.Series(values, index=index, name=input_dataframe.name)
                    by_dataframe = cls.create_backend_frame(index, index=None, columns=None, dtype=None, copy=None)
                    is_mapping = True
        else:
            raise TypeError(f"'{by.__class__.__name__}' object is not callable")

        from ..groupby import DataFrameGroupBy, SeriesGroupBy

        if isinstance(input_dataframe, mpd.DataFrame):
            result = DataFrameGroupBy(
                dataframe=input_dataframe,
                by=by,
                axis=axis,
                level=level,
                as_index=as_index,
                sort=sort,
                group_keys=group_keys,
                squeeze=squeeze,
                observed=observed,
                dropna=dropna,
                by_names=by_names,
                by_dataframe=by_dataframe,
                drop=drop,
                is_mapping=is_mapping,
                default_to_pandas=default_to_pandas
            )
        else:
            result = SeriesGroupBy(
                dataframe=input_dataframe,
                by=by,
                axis=axis,
                level=level,
                as_index=as_index,
                sort=sort,
                group_keys=group_keys,
                squeeze=squeeze,
                observed=observed,
                dropna=dropna,
                by_names=by_names,
                by_dataframe=by_dataframe,
                drop=drop,
                is_mapping=is_mapping,
                default_to_pandas=default_to_pandas
            )
        return result

    @classmethod
    def stat_op(cls, input_dataframe, op_name, axis=0, **kwargs):
        """Compiling statistical operations"""
        stat_functions = {
            "max": {"map_func": ff.max, "reduce_func": ff.max},
            "min": {"map_func": ff.min, "reduce_func": ff.min},
            "median": {"reduce_func": ff.median},
            "mean": {"map_func": ff.sum_count, "reduce_func": ff.reduce_mean},
            "count": {"reduce_func": ff.count},
            "sum": {"map_func": ff.sum_count, "reduce_func": ff.reduce_sum},
            "std": {"reduce_func": ff.std},
            "var": {"reduce_func": ff.var},
            "prod": {"reduce_func": ff.prod}
        }

        if op_name not in stat_functions:
            raise NotImplementedError("Operation not supported")

        functions = stat_functions[op_name]
        if "map_func" in functions:
            map_func = functions["map_func"](axis=axis, **kwargs)
            reduce_func = functions["reduce_func"](axis=axis, **kwargs)
            result = input_dataframe.backend_frame.map_reduce(map_func=map_func, reduce_func=reduce_func, axis=axis,
                                                              concat_axis=axis ^ 1)
        else:
            reduce_func = functions["reduce_func"](axis=axis, **kwargs)
            result = input_dataframe.backend_frame.reduce(func=reduce_func, axis=axis)

        result = mpd.Series(data=result)
        if isinstance(input_dataframe, mpd.Series):
            result = result.squeeze()
        return result

    @classmethod
    def logical_op(cls, input_dataframe, op_name, axis, bool_only, skipna, level, **kwargs):
        """Compiling logical operation"""
        func = getattr(ff, op_name)(axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs)

        if level is None and bool_only is not True:
            result = input_dataframe.backend_frame.map_reduce(map_func=func, reduce_func=func, axis=axis,
                                                              concat_axis=axis ^ 1)
        else:
            result = input_dataframe.backend_frame.reduce(func=func, axis=axis)

        if level is not None:
            return type(input_dataframe)(result)

        result = mpd.Series(data=result)
        if isinstance(input_dataframe, mpd.Series):
            result = result.squeeze()
        return result

    @classmethod
    def math_op(cls, df, op, other, axis='columns', level=None, fill_value=None):
        """Compiling math operators"""
        func = ff.math_op(op, axis, level, fill_value)
        if is_scalar(other):
            frame = df.backend_frame.injective_map(None, other, func, is_scalar=True)
        else:
            if isinstance(other, pandas.DataFrame):
                other = mpd.DataFrame(data=other)
            if isinstance(other, pandas.Series):
                if other.name != df.name:
                    other = mpd.Series(data=other, name=df.name)
                    other.name = df.name = None
                else:
                    other = mpd.Series(data=other)
            if isinstance(other, mpd.Series):
                if other.name != df.name:
                    other.name = df.name = None
            frame = df.backend_frame.injective_map_with_join(func, other.backend_frame)

        if isinstance(df, mpd.DataFrame):
            return mpd.DataFrame(data=frame)

        return mpd.Series(data=frame)

    @classmethod
    def from_numpy(cls, input_array):
        """
        Function of create MindPandas DataFrame from numpy
        This function is not called from any external pandas API.
        """
        frame = BaseIO.from_numpy(input_array)
        return mpd.DataFrame(data=frame)

    @classmethod
    def from_pandas(cls, pandas_input):
        """
        Function of create MindPandas DataFrame from numpy
        This function is not called from any external pandas API.
        """
        frame = BaseIO.from_pandas(pandas_input)
        return mpd.DataFrame(data=frame)

    @classmethod
    def groupby_drop(cls, input_dataframe, index=None, columns=None):
        """
        Drop the unused columns or rows for compiling groupby query
        """
        if index is not None:
            index_numeric_indices = np.sort(input_dataframe.index.get_indexer_for(
                input_dataframe.index[~input_dataframe.index.isin(index)].unique()))
        else:
            index_numeric_indices = None
        if columns is not None:
            columns_numeric_indices = np.sort(input_dataframe.columns.get_indexer_for(
                input_dataframe.columns[~input_dataframe.columns.isin(columns)].unique()))
        else:
            columns_numeric_indices = None
        mask_func = ff.mask_iloc()
        frame = input_dataframe.backend_frame.view(rows_index_numeric=index_numeric_indices,
                                                   columns_index_numeric=columns_numeric_indices,
                                                   func=mask_func)
        return frame

    @classmethod
    def groupby_reduce(cls, groupby_method_name, input_dataframe, **kwargs):
        """
        The reduce algebra used for compiling a DataFrameGroupby or SeriesGroupby instance
        args:
            groupby_method_name: the function application name of groupby.
            input_dataframe: a DataFrameGroupBy or SeriesGroupby.
            kwargs: dict, keyword parameters passed to groupby method.
        """
        drop = input_dataframe.drop
        axis = input_dataframe.axis
        by_dataframe = input_dataframe.by_dataframe
        by_names = input_dataframe.by_names
        is_mapping = input_dataframe.is_mapping
        get_item_key = input_dataframe.get_item_key

        # input_dataframe: DataFrameGroupBy type
        kwargs['groupby_kwargs'] = input_dataframe.kwargs  # add constructor parameters into kwargs
        kwargs['is_mapping'] = is_mapping
        mask_func = ff.mask_iloc()

        if isinstance(input_dataframe, mpd.SeriesGroupBy) and get_item_key is not None:
            index = input_dataframe.by_dataframe.to_pandas()
            if isinstance(index, pandas.DataFrame):
                index = pandas.MultiIndex.from_frame(index)
            else:
                index = pandas.Index(index)

            new_dataframe = mpd.Series(input_dataframe.backend_frame.get_columns(get_item_key, func=mask_func))
            new_dataframe.index = index

            input_dataframe = mpd.SeriesGroupBy(
                new_dataframe,
                drop=False,
                **kwargs,
            )

        if groupby_method_name == "size":
            agg_func = {input_dataframe.columns[0]: "size"}
            map_func, reduce_func = get_groupby_reduce_dict_functions(agg_func, by_names, **kwargs)
        else:
            map_func, reduce_func = get_groupby_reduce_functions(groupby_method_name, by_names, **kwargs)
        reset_index_func = ff.reset_index()

        apply_indices = list(map_func.keys()) if isinstance(map_func, dict) else None

        # by_dataframe restore the order by by_names
        # NOTE: Need to speed up here & move into inner
        if by_names is not None:
            by_names = by_names if isinstance(by_names, (list, pandas.Index)) else [by_names]
            if all(by_names != by_dataframe.columns):  # if the order is not correct, then restore the order
                order_by_dataframe = by_dataframe.to_pandas().reindex(columns=by_names)
                by_dataframe = cls.create_backend_frame(order_by_dataframe, index=None, columns=None, dtype=None,
                                                        copy=None)

        if axis == 0:  # The value in argument `drop` is passed into `groupby_reduce()`.
            frame = input_dataframe.backend_frame.groupby_reduce(axis, by_dataframe, drop, map_func, reduce_func,
                                                                 reset_index_func, apply_indices)
        elif drop:  # and axis == 1 or grouped by column(s)
            frame = input_dataframe.backend_frame.groupby_reduce(axis, by_dataframe, drop, map_func, reduce_func,
                                                                 reset_index_func, apply_indices)
        else:
            frame = input_dataframe.backend_frame.groupby_reduce(axis, by_dataframe, drop, map_func, reduce_func,
                                                                 reset_index_func, apply_indices)

        # Aggregate function `size()` needs a special handle. The output contains only the column name "size".
        # If the `groupby`'s argument as_index is set to False, then the output will include the grouping column.
        # For example:
        # if the input dataframe is

        # df = pd.DataFrame({'c1': [1, 2, 1, 3]})

        # The output of `df.groupby(by='c1').size()` is a Series shown below

        # c1
        # 1  2
        # 2  1
        # 3  1
        # dtype: int64

        # The output of `df.groupby(by='c1', as_index=False).size()` is a DataFrame shown below

        #    c1  size
        # 0   2     2
        # 1   4     2
        # 2   6     2

        squeeze = False
        if groupby_method_name == 'size':
            if drop:
                squeeze = True
            else:
                map_func = ff.rename(columns={0: 'size'})
                frame = frame.reduce(map_func, axis=1)

        if isinstance(input_dataframe, mpd.groupby.SeriesGroupBy) or squeeze:
            result = mpd.Series(data=frame.squeeze())
        else:
            result = mpd.DataFrame(data=frame)

        return result

    @classmethod
    def abs(cls, input_dataframe):
        map_func = ff.abs()
        frame = input_dataframe.backend_frame.map(map_func)
        return type(input_dataframe)(frame)

    @classmethod
    def fillna(cls, **kwargs):
        """Compiling fillna"""
        input_dataframe = kwargs.pop("input_dataframe", None)
        squeeze_self = kwargs.get("squeeze_self", False)
        axis = kwargs.get("axis", 0)
        value = kwargs.pop("value")
        method = kwargs.get("method", None)
        limit = kwargs.get("limit", None)
        full_axis = method is not None or limit is not None

        if isinstance(value, dict):
            if squeeze_self:
                map_func = ff.fill_na(value=value, **kwargs)
            else:
                func_dict = {
                    col: val for (col, val) in value.items() if col in input_dataframe.columns
                }
                map_func = ff.fill_na(value=func_dict, **kwargs)
        else:
            map_func = ff.fill_na(value=value, **kwargs)

        if full_axis:
            frame = input_dataframe.backend_frame.reduce(func=map_func, axis=axis)
        else:
            frame = input_dataframe.backend_frame.map(map_func)
        if isinstance(input_dataframe, mpd.Series):
            return mpd.Series(data=frame)
        return mpd.DataFrame(data=frame)

    @classmethod
    def rename(cls, input_dataframe, mapper=None, *, index=None, columns=None, axis=None,
               copy=True, inplace=False, level=None, errors='ignore'):
        """Compiling rename"""
        if columns is not None or index is not None:
            map_func = ff.rename(mapper=mapper, index=index, columns=columns, axis=axis,
                                 copy=copy, inplace=inplace, level=level, errors=errors)

            if columns is not None:
                frame = input_dataframe.backend_frame.reduce(map_func, axis=1)
            elif index is not None:
                frame = input_dataframe.backend_frame.reduce(map_func, axis=0)

            return mpd.DataFrame(data=frame)

        map_func = ff.rename(mapper=mapper, index=index, columns=columns, axis=axis,
                             copy=copy, level=level, errors=errors)

        frame = input_dataframe.backend_frame.map(map_func)

        if inplace:
            input_dataframe.backend_frame = frame
            return None
        return mpd.DataFrame(data=frame)

    @classmethod
    def isna(cls, **kwargs):
        """Compiling isna"""
        input_dataframe = kwargs.pop("input_dataframe", None)
        is_series = kwargs.pop("is_series", False)
        func = ff.isna(**kwargs)
        frame = input_dataframe.backend_frame.map(func)
        if is_series:
            return mpd.Series(data=frame)

        return mpd.DataFrame(data=frame)

    @classmethod
    def dtypes(cls, **kwargs):
        """Compiling dtypes"""
        input_dataframe = kwargs.pop("input_dataframe", None)
        func = ff.dtypes(**kwargs)
        frame = input_dataframe.backend_frame.map(func)
        output_frame = frame.reduce(ff.dtypes_post(), 0, 1)
        result = mpd.Series(output_frame)
        if isinstance(input_dataframe, mpd.DataFrame):
            return result

        return result[0]

    @classmethod
    def isin(cls, input_dataframe, **kwargs):
        """Compiling isin"""
        func = ff.isin(**kwargs)
        frame = input_dataframe.backend_frame.map(func)
        return type(input_dataframe)(data=frame)

    @classmethod
    def dropna(cls, input_dataframe, axis, how, thresh, subset, inplace):
        """Compiling dropna"""
        func = ff.dropna(axis, how, thresh, subset, inplace)
        # Change axis since reduce uses axis of 0 as column
        frame = input_dataframe.backend_frame.reduce(func, axis=axis ^ 1)
        return mpd.DataFrame(data=frame)

    @classmethod
    def drop(cls, input_dataframe, index=None, columns=None, inplace=False, ignore_index=False):
        """Compiling drop"""
        map_func = ff.drop(index=index, columns=columns)
        frame = input_dataframe.backend_frame.map(map_func=map_func)
        frame = frame.validate_partitions()
        result = mpd.DataFrame(frame)
        if ignore_index:
            result = result.reset_index(drop=True)
        if inplace:
            input_dataframe.set_backend_frame(result.backend_frame)
            return None
        return result

    @classmethod
    def deprecated_drop(cls, input_dataframe, index=None, columns=None, ignore_index=False):
        """deprecated function: Compiling drop"""
        if index is not None:
            index_numeric_indices = np.sort(input_dataframe.index.get_indexer_for(
                input_dataframe.index[~input_dataframe.index.isin(index)].unique()
            ))
        else:
            index_numeric_indices = np.sort(input_dataframe.index.get_indexer_for(input_dataframe.index))
        if columns is not None:
            columns_numeric_indices = np.sort(input_dataframe.columns.get_indexer_for(
                input_dataframe.columns[~input_dataframe.columns.isin(columns)].unique()
            ))
        else:
            columns_numeric_indices = np.sort(input_dataframe.columns.get_indexer_for(
                input_dataframe.columns
            ))
        mask_func = ff.mask_iloc()
        frame = input_dataframe.backend_frame.view(rows_index_numeric=index_numeric_indices,
                                                   columns_index_numeric=columns_numeric_indices,
                                                   func=mask_func)

        result = mpd.DataFrame(frame)
        if ignore_index:
            result = result.reset_index(drop=True)
        return result

    @classmethod
    def getitem_array(cls, input_dataframe, key):
        """
        Compiling getitem for the case when getitem's result is an array.
        This function is not called from any external pandas API.
        """
        if isinstance(key, slice):
            return cls.getitem_row_array(input_dataframe, key)
        if isinstance(key, mpd.Series):
            get_item_fn = ff.get_item(is_scalar=True)
            df_row_split_points = input_dataframe.backend_frame.get_axis_split_points(axis=0)
            key_row_split_points = key.backend_frame.get_axis_split_points(axis=0)
            if not np.array_equal(df_row_split_points, key_row_split_points):
                key = key.backend_frame.axis_repartition(axis=0, mblock_size=i_config.get_min_block_size(),
                                                         by='split_pos', by_data=df_row_split_points)
                frame = input_dataframe.backend_frame.injective_map(key, None, get_item_fn, is_scalar=True)
            else:
                frame = input_dataframe.backend_frame.injective_map(key.backend_frame, None, get_item_fn,
                                                                    is_scalar=True)
            # delete the empty dataframe if any exists in the output_partitions
            frame = frame.remove_empty_rows()
            return type(input_dataframe)(frame)

        index = input_dataframe.backend_frame.index
        columns = input_dataframe.backend_frame.columns
        if is_bool_indexer(key) or is_boolean_array(key):
            if isinstance(key, pandas.Series) and not key.index.equals(index):
                warnings.warn(
                    "Boolean Series key will be reindexed to match DataFrame index.",
                    PendingDeprecationWarning,
                    stacklevel=3)
            elif len(index) != len(key):
                raise ValueError(f"Item wrong length {len(key)} instead of {len(index)}.")
            key = index[check_bool_indexer(index, key)]
            if key.size:
                return cls.getitem_row_array(input_dataframe, key)

            # Return an empty DataFrame of all the input columns
            return cls.from_pandas(pandas.DataFrame(columns=columns))

        if not is_list_like(key, list) and key not in columns:
            raise KeyError(f"{key} not in index")

        if any(k not in columns for k in key):
            # If this code is change, please update the error msg in `project_with_pred`.
            msg = str([k for k in key if k not in columns]).replace(",", "")
            raise KeyError(f"{msg} not in index")
        return cls.getitem_column_array(input_dataframe, key)

    @classmethod
    def project_with_pred(cls, df, columns, lhs: str, op: str, rhs: int):
        """
        Compiling project_with_pred.
        This function is not called from any external pandas API.
        """
        df_columns = df.columns
        mask_func = ff.mask_iloc()
        if isinstance(columns, (list, np.ndarray, pandas.Index, pandas.Series, mpd.Series)):
            output_series = False
            # Flag an error if the projected column(s) are not in the input dataframe
            # This is in line with the error raised in `getitem_array`.
            if any(c not in df_columns for c in columns):
                msg = str([c for c in columns if c not in df_columns]).replace(",", "")
                raise KeyError(f"{msg} not in index")
        else:
            output_series = True
            # Flag an error if the projected column is not in the input dataframe
            # This is in line with the error raised in `getitem_column`.
            if columns not in df.columns:
                raise AttributeError(f"{type(df).__name__} object has no attribute '{columns}'")
            columns = [columns]
        if output_series:
            frame = df.backend_frame.get_columns(columns, lhs, op, rhs, func=mask_func)
            return mpd.Series(frame)
        frame = df.backend_frame.get_columns(columns, lhs, op, rhs, func=mask_func)
        return mpd.DataFrame(frame)

    @classmethod
    def getitem_column(cls, input_dataframe, key):
        """
        Compiling getitem for the case when getitem's result is a column.
        This function is not called from any external pandas API.
        """
        key = apply_if_callable(key, input_dataframe)
        if key not in input_dataframe.columns or cls.has_multiindex(input_dataframe, axis=1):
            # If this code is change, please update the error msg in `project_with_pred`.
            raise AttributeError(f"{type(input_dataframe).__name__} object has no attribute '{key}'")
        if isinstance(key, (str, int)):
            is_series = True
            key_lst = [key]
        else:
            is_series = False
        ser = cls.getitem_column_array(input_dataframe, key_lst, is_series)
        # Does the squeeze on the column axis always results in a Series?
        if isinstance(ser, mpd.Series):
            ser.parent = input_dataframe
            ser.parent_axis = 1
            ser.parent_key = key_lst
            ser.name = key
        return ser

    @classmethod
    def getitem_column_array(cls, input_dataframe, columns=None, is_series=False):
        """
        Compiling getitem for the case when getitem's result is a column array.
        This function is not called from any external pandas API.
        """
        mask_func = ff.mask_iloc()
        frame = input_dataframe.backend_frame.get_columns(columns, func=mask_func)
        if is_series:
            return mpd.Series(frame)
        if is_scalar(columns) and hasattr(frame, 'columns') and frame.series_like:
            return mpd.Series(frame)
        return mpd.DataFrame(frame)

    @classmethod
    def getitem_row_array(cls, input_dataframe, indexes=None, indices_numeric=False):
        """
        Compiling getitem for the case when getitem's result is a row array.
        This function is not called from any external pandas API.
        """
        mask_func = ff.mask_iloc()
        frame = input_dataframe.backend_frame.get_rows(indexes, indices_numeric, func=mask_func)
        input_type = type(input_dataframe)
        return input_type(frame)

    @classmethod
    def getitem_row_array_with_pred(cls, input_dataframe, columns, lhs: str, op: str, rhs: int):
        """
        Compiling getitem for the case when getitem's result is a row array with pred.
        This function is not called from any external pandas API.
        """
        if isinstance(columns, list):
            output_series = False
        else:
            output_series = True
            columns = [columns]
        frame = input_dataframe.backend_frame.get_rows_with_pred(columns, lhs, op, rhs)
        if output_series:
            return mpd.Series(frame)

        return mpd.DataFrame(frame)

    @classmethod
    def append(cls, input_dataframe, other, ignore_index=False, verify_integrity=False, sort=False):
        """Compiling append"""
        append_df = input_dataframe.backend_frame.append(other.backend_frame)
        output_df = mpd.DataFrame(append_df)

        if verify_integrity:
            list_input = list(input_dataframe.index)
            list_other = list(other.index)
            for i in range(len(list_input)):
                for j in range(len(list_other)):
                    if list_input[i] == list_other[j]:
                        raise ValueError("creating index with duplicates.")

        if ignore_index:
            output_df = cls.reset_index(output_df, drop=True)

        if sort:
            output_df = output_df.sort_index(axis=1)

        return output_df

    @classmethod
    def merge(cls, **kwargs):
        """Compiling merge"""
        how = kwargs.get('how')
        on = kwargs.get('on')
        left_on = kwargs.get('left_on')
        right_on = kwargs.get('right_on')
        suffixes = kwargs.get('suffixes')
        left = kwargs.pop('left')
        right = kwargs.pop('right')
        sort = kwargs.pop('sort')

        if how in ['left', 'inner']:
            apply_func = ff.merge(right, **kwargs)
            frame = left.backend_frame.reduce(apply_func, axis=1)
        else:
            apply_func = ff.merge(left, **kwargs)
            frame = right.backend_frame.reduce(apply_func, axis=1)

        pandas_frame = frame.to_pandas()
        columns = pandas_frame.columns
        if on:
            if how == 'right':
                by = [o if o in columns else o + suffixes[1] for o in on]
            else:
                by = [o if o in columns else o + suffixes[0] for o in on]
        elif left_on and right_on:
            if how == 'right':
                by = [o if o in columns else o + suffixes[1] for o in right_on]
            else:
                by = [o if o in columns else o + suffixes[0] for o in left_on]
        else:
            by = columns

        if sort:
            pandas_frame.sort_values(
                ignore_index=True,
                kind="mergesort",
                by=by,
                inplace=True
            )
        else:
            if how == 'inner':
                key = left.backend_frame.to_pandas().loc[:, by[0]]
                key.drop_duplicates(inplace=True)
                key.reset_index(inplace=True, drop=True)
                lkey = {v: k for k, v in key.to_dict().items()}
                pandas_frame.sort_values(
                    ignore_index=True,
                    kind="mergesort",
                    by=by,
                    inplace=True,
                    key=lambda x: x.map(lkey)
                )
            else:
                pandas_frame.reset_index(inplace=True, drop=True)

        return mpd.DataFrame(pandas_frame)

    @classmethod
    def create_backend_frame(cls, data, index, columns, dtype, copy, container_type=pandas.DataFrame):
        """Function of create the backend frame"""
        frame = BaseIO.create_backend_frame(data, index, columns, dtype, copy, container_type)
        return frame

    @classmethod
    def default_to_pandas(cls, df, df_method, *args, force_series=False, **kwargs):
        """Default the API call to pandas.
        Args:
            df: mindpandas.DataFrame.
            df_method: callable, the API that is called.
            args: tuple, parameters passed to df_method.
            force_series: bool, force to default to use pandas.Series.
            kwargs: dict, keyword parameters passed to df_method.
        Returns:
            result: mindpandas.DataFrame.
        Raises:
            AttributeError: when there is no corresponding API found in pandas.
        """
        pandas_df = df.backend_frame.to_pandas(force_series=force_series)
        func_name = getattr(df_method, "__name__", str(df_method))

        list_args = [args] if not is_list_like(args) else args
        if isinstance(list_args, tuple):
            list_args = list(list_args)
        for i, arg in enumerate(list_args):
            if hasattr(arg, "to_pandas"):
                list_args[i] = arg.to_pandas()
        args = tuple(list_args)

        for param_name, param in kwargs.items():
            if hasattr(param, "to_pandas"):
                kwargs[param_name] = param.to_pandas()

        try:
            if func_name in ['loc', 'iloc']:
                assert len(args) == 3
                getattr(pandas_df, func_name)[args[1], args[0]] = args[2]
                df.backend_frame = df.backend_frame.update(pandas_df)
                return None

            if func_name == 'agg':
                args_ = list(args)
                # Second argument to agg must be axis
                if 'agg_axis' in kwargs:
                    agg_axis = kwargs['agg_axis']
                    del kwargs['agg_axis']
                else:
                    agg_axis = 0
                args_.insert(1, agg_axis)
                args = tuple(args_)

            result = getattr(pandas_df, func_name)(*args, **kwargs)

            # setitem does the inplace changes
            if func_name in ['__setitem__']:
                df.backend_frame = df.backend_frame.update(pandas_df)
                return None

            if isinstance(result, tuple):
                return tuple(cls.try_cast_to_mindpandas(data) for data in result)

            return cls.try_cast_to_mindpandas(result)
        except AttributeError as err:
            raise RuntimeError(f"{func_name} not supported") from err

    @classmethod
    def try_cast_to_mindpandas(cls, data):
        """
        Function to cast pandas.DataFrame or pandas.Series to
        mindpandas.DataFrame or mindpandas.Series.
        """
        if isinstance(data, pandas.DataFrame):
            return mpd.DataFrame(data)
        if isinstance(data, pandas.Series):
            return mpd.Series(data)
        return data

    @classmethod
    def groupby_default_to_pandas(cls, input_dataframe, groupby_method_name, force_series=False, **kwargs):
        """Default the groupby API call to pandas.
        Args:
            input_dataframe: mindpandas.DataFrame.
            groupby_method_name: callable, the API that is called.
            force_series: bool, parameters passed to determine if use Series.groupby.
            kwargs: dict, keyword parameters passed to groupby_method_name.
        Returns:
            result: mindpandas.DataFrame or mindpandas.Series.
        Raises:
            AttributeError: when there is no corresponding API found in pandas.
        """
        by = input_dataframe.by
        axis = input_dataframe.axis
        get_item_key = input_dataframe.get_item_key
        level = input_dataframe.kwargs['level']
        sort = input_dataframe.kwargs['sort']
        group_keys = input_dataframe.kwargs['group_keys']
        observed = input_dataframe.kwargs['observed']
        dropna = input_dataframe.kwargs['dropna']

        def process_by(by):
            if isinstance(by, list) and all(isinstance(n, mpd.Series) for n in by):
                by = [x.to_pandas() for x in by]
            elif isinstance(by, mpd.Series):
                by = by.to_pandas()
            return by

        by = process_by(by)

        if not force_series or get_item_key is None:
            pandas_groupbyframe = input_dataframe.backend_frame \
                .to_pandas(force_series=force_series) \
                .groupby(by, axis=axis,
                         level=level,
                         as_index=input_dataframe.drop,
                         sort=sort,
                         group_keys=group_keys,
                         observed=observed,
                         dropna=dropna)
        else:
            pandas_groupbyframe = input_dataframe.backend_frame \
                .to_pandas(force_series=False) \
                .groupby(by, axis=axis,
                         level=level,
                         as_index=input_dataframe.drop,
                         sort=sort,
                         group_keys=group_keys,
                         observed=observed,
                         dropna=dropna)

        if get_item_key is not None:
            if isinstance(get_item_key, mpd.Series):
                get_item_key = get_item_key.to_pandas()
            pandas_groupbyframe = pandas_groupbyframe[get_item_key]

        for param_name, param in kwargs.items():
            if hasattr(param, "backend_frame"):
                kwargs[param_name] = param.backend_frame.to_pandas()

        if groupby_method_name == 'agg':
            result = getattr(pandas_groupbyframe, 'agg')(**kwargs)
            if isinstance(result, pandas.Series):
                return mpd.Series(result)
            return mpd.DataFrame(result)

        try:
            groupby_attr = getattr(type(pandas_groupbyframe), groupby_method_name)
        except AttributeError as err:
            raise RuntimeError(f"{groupby_method_name} not supported") from err

        if isinstance(groupby_attr, property):
            property_attr = getattr(pandas_groupbyframe, groupby_method_name)
            result = property_attr(**kwargs) if callable(property_attr) else property_attr
        else:
            result = groupby_attr(pandas_groupbyframe, **kwargs)

        if groupby_method_name == 'resample':
            # By default pandas does not return a grouper for resample op
            # And expects a further reduction operation such as sum()
            return result

        if isinstance(result, pandas.Series):
            return mpd.Series(result)
        if isinstance(result, pandas.DataFrame):
            if result.index.name in result.columns:
                result = result.drop(result.index.name, axis=1)
            return mpd.DataFrame(result)
        return result

    @classmethod
    def default_to_pandas_general(cls, pandas_method, *args, **kwargs):
        """Default the API call to pandas.
        Args:
            pandas_method: callable, the API that is called.
            args: tuple, parameters passed to df_method.
            kwargs: dict, keyword parameters passed to df_method.
        Returns:
            result: mindpandas.DataFrame or pandas.DatetimeIndex
        Raises:
            AttributeError: when there is no corresponding API found in pandas.
        """
        func_name = getattr(pandas_method, "__name__", str(pandas_method))

        list_args = list(args)
        for i, arg in enumerate(list_args):
            if hasattr(arg, "to_pandas"):
                list_args[i] = arg.to_pandas()
        args = tuple(list_args)

        for param_name, param in kwargs.items():
            if hasattr(param, "to_pandas"):
                kwargs[param_name] = param.to_pandas()

        try:
            result = getattr(pandas, func_name)(*args, **kwargs)
        except AttributeError as err:
            raise RuntimeError(f"{func_name} not supported") from err

        if not isinstance(result, (pandas.DataFrame, pandas.Series)):
            return result
        return mpd.DataFrame(result)

    @classmethod
    def combine(cls, input_dataframe, other_dataframe, orig_df_cols, orig_other_cols, df_shapes, func, fill_value=None,
                overwrite=True, **kwargs):
        """Compiling combine"""
        fold_func = ff.combine(func=func,
                               fill_value=fill_value,
                               overwrite=overwrite,
                               **kwargs)
        frame = input_dataframe.backend_frame.combine_fold(other_dataframe.backend_frame, orig_df_cols, orig_other_cols,
                                                           fold_func, df_shapes)
        return mpd.DataFrame(data=frame)

    @classmethod
    def explode(cls, input_dataframe, column, ignore_index=False):
        """
        Compiling explode
        """
        reduce_func = ff.explode(column=column, ignore_index=ignore_index)
        result = input_dataframe.backend_frame.reduce(func=reduce_func, axis=1)
        return mpd.DataFrame(result)

    @classmethod
    def pivot_table(cls,
                    data,
                    values,
                    index,
                    columns,
                    aggfunc,
                    fill_value,
                    margins: bool,
                    dropna: bool,
                    margins_name: str,
                    observed: bool,
                    sort: bool,
                    ):
        """
        Based on pivot_table from pandas, with modifications:
        (1) Use parallelized groupby,
        (2) Aggregrate grouped columns using only the
            available mindpandas DataFrameGroupBy reduction
            methods,
        (3) Convert to pandas for handling fp accuracy, margins,
            MultiIndex
        """

        keys = index + columns

        values_passed = values is not None
        if values_passed:
            if is_list_like(values):
                values_multi = True
                values = list(values)
            else:
                values_multi = False
                values = [values]

            # Make sure value labels are in data
            for i in values:
                if i not in data:
                    raise KeyError(i)

            to_filter = []
            for x in keys + values:
                if isinstance(x, pandas.Grouper):
                    x = x.key
                try:
                    if x in data:
                        to_filter.append(x)
                except TypeError:
                    pass
            if len(to_filter) < len(data.columns):
                data = data[to_filter]

        else:
            values = data.columns
            for key in keys:
                try:
                    values = values.drop(key)
                except (TypeError, ValueError, KeyError):
                    pass
            values = list(values)

        if isinstance(aggfunc, dict):
            # Requires DataFrameGroupBy.agg
            data = data.to_pandas()

            grouped = data.groupby(keys, observed=observed, sort=sort)
            agged = grouped.agg(aggfunc)
            agged = mpd.DataFrame(agged)

        else:
            # Use our parallel groupby
            grouped = data.groupby(keys, observed=observed, sort=sort)

            try:
                groupby_attr = getattr(grouped, aggfunc)
            except AttributeError as err:
                raise RuntimeError(f"{aggfunc} not supported") from err

            # Apply aggregation function on grouped dataframe
            agged = groupby_attr()

        agged_pandas = None
        if (
                isinstance(agged, mpd.DataFrame)
                and dropna
                and agged.columns.size
        ):
            agged = agged.dropna(how="all")

            if isinstance(data, mpd.DataFrame):
                data_pandas = data.to_pandas()
            else:
                data_pandas = data

            if isinstance(agged, mpd.DataFrame):
                agged_pandas = agged.to_pandas()
            else:
                agged_pandas = agged

            for v in values:
                if v in data_pandas and v in agged_pandas:
                    if is_integer_dtype(data_pandas[v]) and not is_integer_dtype(agged_pandas[v]):
                        if isinstance(agged_pandas[v], ABCDataFrame):
                            # To match Pandas v1.3.3 output, this
                            # op does the necessary type casting
                            agged_pandas[v] = agged_pandas[v]
                        else:
                            agged_pandas[v] = maybe_downcast_to_dtype(agged_pandas[v], data_pandas[v].dtype)

        if agged_pandas is not None:
            agged = mpd.DataFrame(agged_pandas)

        table = agged

        # Unstack the upper index levels
        if table.index.nlevels > 1 and index:
            index_names = agged.index.names[: len(index)]
            to_unstack = []
            for i in range(len(index), len(keys)):
                name = agged.index.names[i]
                if name is None or name in index_names:
                    to_unstack.append(i)
                else:
                    to_unstack.append(name)
            # Requires assignment to agged to match pandas output
            table = agged.unstack(to_unstack)

        if not dropna:
            if isinstance(table.index, MultiIndex):
                m = MultiIndex.from_arrays(
                    cartesian_product(table.index.levels), names=table.index.names
                )
                if isinstance(table, mpd.DataFrame):
                    # MultiIndex not supported for mpd.DataFrame.reindex
                    table = table.to_pandas()
                table = table.reindex(m, axis=0)

            if isinstance(table.columns, MultiIndex):
                m = MultiIndex.from_arrays(
                    cartesian_product(table.columns.levels), names=table.columns.names
                )
                if isinstance(table, mpd.DataFrame):
                    # MultiIndex not supported for mpd.DataFrame.reindex
                    table = table.to_pandas()
                table = table.reindex(m, axis=1)

            if isinstance(table, ABCDataFrame):
                table = mpd.DataFrame(table)

        if isinstance(table, mpd.DataFrame):
            table = table.sort_index(axis=1)

        if fill_value is not None:
            table = table.fillna(fill_value, downcast="infer")

        if margins:
            if isinstance(data, mpd.DataFrame):
                data = data.to_pandas()
            if isinstance(table, mpd.DataFrame):
                table = table.to_pandas()

            if dropna:
                data = data[data.notna().all(axis=1)]

            table = _add_margins(table,
                                 data,
                                 values,
                                 rows=index,
                                 cols=columns,
                                 aggfunc=aggfunc,
                                 observed=dropna,
                                 margins_name=margins_name,
                                 fill_value=fill_value,
                                 )
            if isinstance(table, ABCDataFrame):
                table = mpd.DataFrame(table)
            elif isinstance(table, ABCSeries):
                table = mpd.Series(table)

        # discard the top level
        if values_passed and not values_multi and table.columns.nlevels > 1:
            table = table.droplevel(0, axis=1)
        if not index and columns:
            table = table.T

        # make sure empty columns are removed if dropna=True
        if isinstance(agged, mpd.DataFrame) and dropna:
            table = table.dropna(how="all", axis=1)

        if isinstance(table, ABCSeries):
            return mpd.Series(table)

        if isinstance(table, ABCDataFrame):
            return mpd.DataFrame(table)

        return table

    @classmethod
    def cum_op(cls, input_dataframe, method, axis, skipna, *args, **kwargs):
        """Compiling cumulative ops."""
        reduce_func = getattr(ff, method)(axis=axis, skipna=skipna, *args, **kwargs)
        result = input_dataframe.backend_frame.reduce(axis=axis, func=reduce_func)
        return type(input_dataframe)(result)

    @classmethod
    def apply(cls, df, **kwargs):
        """Compiling apply"""
        if isinstance(df, mpd.Series):
            axis = kwargs.pop('axis')
        else:
            axis = kwargs.get('axis')
        if "apply_from_agg" in kwargs:
            apply_from_agg = kwargs.pop("apply_from_agg")
        else:
            apply_from_agg = 0
        func = kwargs.get('func')
        if isinstance(func, dict):
            return cls.default_to_pandas(df, df_method=cls.apply, **kwargs)

        # Handle a case from df.agg where a list of functions is passed and
        # the result is a series-like dataframe, but Pandas does not squeeze it
        do_not_squeeze = False
        if apply_from_agg and isinstance(func, list) and len(func) > 1:
            do_not_squeeze = True
        kwargs.pop('func')
        validated_func = cls._validate_and_get_func(func)
        wrapped_func = ff.apply(validated_func, **kwargs)
        result = df.backend_frame.reduce(wrapped_func, axis=axis)
        if not do_not_squeeze and (isinstance(df, mpd.Series) or result.series_like):
            return mpd.Series(result)

        return mpd.DataFrame(result)

    @classmethod
    def _validate_and_get_func(cls, func):
        """
        Validate func type and get function reference if a string is given.
        Args:
            func: the function passed by the user.
        Returns:
            func: [callable], validated functions.
        Raises:
            TypeError: when the type of func is not valid.
        """
        if isinstance(func, str):
            return cls._validate_string_func(func)
        if isinstance(func, list):
            return cls._validate_list_func(func)
        if callable(func):
            return func

        raise TypeError(f"function type {type(func)} not supported")

    @classmethod
    def _validate_string_func(cls, func):
        """
        Validate string if it is a func and get function reference.
        Args:
            func: the string passed by the user.
        Returns:
            func: [callable], validated functions.
        Raises:
            TypeError: when string is not a valid function.
        """
        # Try get function by name from pandas.DataFrame.
        if not hasattr(pandas.DataFrame, func) and not hasattr(np, func):
            raise AttributeError(f"{func} is not a valid function")

        # Unknown function name.
        return func

    @classmethod
    def _validate_list_func(cls, func):
        """
        Validate list of func and get function reference.
        Args:
            func: the list passed by the user.
        Returns:
            func: [callable], validated functions.
        Raises:
            TypeError: when any func type in the list is not valid.
        """
        for i, fn in enumerate(func):
            if isinstance(fn, str):
                func[i] = cls._validate_string_func(fn)
            elif not callable(fn):
                raise TypeError(f"function type {type(fn)} not supported, should be a str or callable object")
        return func

    @classmethod
    def insert(cls, df, **kwargs):
        """Compiling insert"""
        loc = kwargs.pop('loc')
        part_loc, internal_loc = df.backend_frame.get_internal_loc(loc, axis=1, allow_append=True)
        insert_func = ff.insert(loc=internal_loc, **kwargs)
        concat_func = ff.concat()
        frame = df.backend_frame.apply_select_indice_axis(insert_func, concat_func, indice=part_loc, axis=0)
        return frame

    @classmethod
    def rank(cls, input_dataframe, **kwargs):
        """Compiling rank"""
        axis = kwargs.get("axis", 0)
        map_func = ff.rank(**kwargs)
        frame = input_dataframe.backend_frame.reduce(
            map_func,
            axis,
        )
        return mpd.DataFrame(data=frame)

    @classmethod
    def view(cls, indexer, row_ids, col_ids, return_df=False):
        """Compiling view"""
        # if len(col_ids) > 1:
        # output is a series if we changed rows or columns to 1
        # if input was already 1, we take type of input
        if return_df:
            return_type = mpd.DataFrame
        else:
            return_type = mpd.Series
        mask_func = ff.mask_iloc()
        frame = indexer.backend_frame.view(rows_index_numeric=row_ids, columns_index_numeric=col_ids, func=mask_func)
        if return_type == mpd.Series and frame.partition_shape[1] != 1:
            frame.transpose()
        return return_type(frame)

    @classmethod
    def to_datetime(cls, input_dataframe, **kwargs):
        """Compiling to_datetime"""
        reduce_fun = ff.to_datetime(**kwargs)
        frame = input_dataframe.backend_frame.reduce(reduce_fun, axis=1)
        return mpd.Series(data=frame)

    @classmethod
    def reindex(cls, input_dataframe, index, columns, copy, **kwargs):
        """Compiling reindex"""
        output_frame = input_dataframe.backend_frame
        if index is not None:
            if not isinstance(index, pandas.Index):
                index = pandas.Index(index)
            if not index.equals(output_frame.index):
                reduce_func = ff.reindex(labels=index, axis=0, **kwargs)
                output_frame = output_frame.reduce(func=reduce_func, axis=0)
        if columns is not None:
            if not isinstance(columns, pandas.Index):
                columns = pandas.Index(columns)
            if not columns.equals(output_frame.columns):
                reduce_func = ff.reindex(labels=columns, axis=1, **kwargs)
                output_frame = output_frame.reduce(func=reduce_func, axis=1)
        if not copy:
            input_dataframe.set_backend_frame(output_frame)
            return None
        return type(input_dataframe)(output_frame)


    @classmethod
    def sort_rows_by_column_values(cls, input_dataframe, by, **kwargs):
        """
        Function for sorting rows based on the column values.
        This function is not called from any external pandas API.
        """
        ignore_index = kwargs.get("ignore_index", False)
        kwargs["ignore_index"] = False
        if not is_list_like(by):
            by = [by]
        cols_to_sort_by = {
            col: input_dataframe[col].to_pandas() for col in by
        }

        if not input_dataframe.index.has_duplicates:
            sort_by = pandas.DataFrame(cols_to_sort_by)
            new_index = sort_by.sort_values(by=by, **kwargs).index
            output_dataframe = input_dataframe.reindex(axis=0, labels=new_index)
            if ignore_index:
                output_dataframe = output_dataframe.reset_index(drop=True)
        else:
            old_index = input_dataframe.index.copy()
            sort_by_with_dup = pandas.DataFrame(cols_to_sort_by, index=old_index)
            sort_by_without_dup = sort_by_with_dup.reset_index(drop=True)

            sorted_index_with_dup = sort_by_with_dup.sort_values(by, **kwargs).index
            sorted_index_without_dup = sort_by_without_dup.sort_values(by, **kwargs).index

            output_dataframe = input_dataframe.reset_index(drop=True).reindex(axis=0, labels=sorted_index_without_dup)
            if ignore_index:
                output_dataframe = output_dataframe.reset_index(drop=True)
            else:
                output_dataframe.index = sorted_index_with_dup
        return output_dataframe

    @classmethod
    def sort_columns_by_row_values(cls, input_dataframe, by, **kwargs):
        return cls.default_to_pandas(input_dataframe, df_method="sort_values", by=by, **kwargs)

    @classmethod
    def sort_values(cls, input_dataframe, by, **kwargs):
        """Compiling sort_values"""
        axis = kwargs.get("axis", 0)
        inplace = kwargs.get("inplace", False)
        kwargs["inplace"] = False

        if axis == 0:
            sorted_dataframe = cls.sort_rows_by_column_values(input_dataframe=input_dataframe, by=by, **kwargs)
        else:
            sorted_dataframe = cls.sort_columns_by_row_values(input_dataframe=input_dataframe, by=by, **kwargs)

        if inplace:
            input_dataframe.set_backend_frame(sorted_dataframe.backend_frame)
            return None
        return sorted_dataframe

    @classmethod
    def concat(cls, objs, axis=0, is_series=False, obj_is_series=None, **kwargs):
        """Compiling concat"""
        backend_df_list = []
        for obj in objs:
            backend_df_list.append(obj.backend_frame)
        concated_frame = BaseGeneral.concat(backend_df_list, axis=axis, obj_is_series=obj_is_series, **kwargs)
        if is_series:
            result = mpd.Series(data=concated_frame)
        else:
            result = mpd.DataFrame(data=concated_frame)
        return result

    @classmethod
    def sort_index(cls, input_dataframe, axis=0, level=None, ascending=True, inplace=False, kind='quicksort',
                   na_position='last', sort_remaining=True, ignore_index=False, key=None):
        """Compiling sort_index"""
        func = ff.sort_index(axis=axis,
                             level=level,
                             ascending=ascending,
                             inplace=inplace,
                             kind=kind,
                             na_position=na_position,
                             sort_remaining=sort_remaining,
                             ignore_index=ignore_index,
                             key=key)
        frame = input_dataframe.backend_frame.reduce(func=func, axis=axis)
        return mpd.DataFrame(data=frame)

    @classmethod
    def set_index(
            cls, input_dataframe, keys: list, drop: bool = True, append: bool = False
    ):
        """Compiling set_index"""
        frame = input_dataframe.backend_frame
        arrays = []
        names = []
        mask_func = ff.mask_iloc()
        if append:
            names = list(frame.index.names)
            if isinstance(frame.index, pandas.MultiIndex):
                for i in range(frame.index.nlevels):
                    arrays.append(frame.index.get_level_values(i))
            else:
                arrays.append(frame.index)

        to_remove = []
        for col in keys:
            if isinstance(col, pandas.MultiIndex):
                for n in range(col.nlevels):
                    arrays.append(col.get_level_values(n))
                names.extend(col.names)
            elif isinstance(col, (pandas.Index, mpd.Series, pandas.Series)):
                # if Index then not MultiIndex (treated above)
                arrays.append(col)
                names.append(col.name)
            elif isinstance(col, (list, np.ndarray)):
                arrays.append(col)
                names.append(None)
            elif isinstance(col, Iterator):
                arrays.append(list(col))
                names.append(None)
                # from here, col can only be a column label
            else:
                to_remove.append(col)
                continue

            if len(arrays[-1]) != len(frame.index):
                # check newest element against length of calling frame, since
                # ensure_index_from_sequences would not raise for append=False.
                raise ValueError(
                    f"Length mismatch: Expected {len(frame.index)} rows, "
                    f"received array of length {len(arrays[-1])}"
                )

        if to_remove:
            extract_index = input_dataframe.backend_frame.to_labels(to_remove, func=mask_func)
            names.extend(extract_index.names)
            if isinstance(extract_index, pandas.MultiIndex):
                for i in range(extract_index.nlevels):
                    arrays.append(extract_index.get_level_values(i))
            else:
                arrays.append(extract_index)

        new_index = ensure_index_from_sequences(arrays, names)

        if drop:
            new_columns_list = [i for i in frame.columns if i not in to_remove]
            result = frame.get_columns(new_columns_list, func=mask_func)
        else:
            new_columns_list = None
            result = frame.copy()

        # lazily mask the frame based on given columns_index
        frame = result.view(columns_index=new_columns_list, func=mask_func)
        frame.index = new_index
        return mpd.DataFrame(data=frame)

    @classmethod
    def squeeze(cls, input_dataframe, axis=None):
        """Compiling squeeze"""
        squeezed_results = input_dataframe.backend_frame.squeeze(axis=axis)
        input_dataframe_index = input_dataframe.backend_frame.index
        input_dataframe_columns = input_dataframe.backend_frame.columns
        if axis is None and (len(input_dataframe_index) == 1 or len(input_dataframe_columns) == 1):
            return mpd.Series(squeezed_results).squeeze()
        if (axis == 1 and len(input_dataframe_columns) == 1) or (axis == 0 and len(input_dataframe_index) == 1):
            return mpd.Series(squeezed_results)

        return input_dataframe

    @classmethod
    def applymap(cls, df, func, na_action, **kwargs):
        """Compiling applymap"""
        map_func = ff.applymap(func=func, na_action=na_action, **kwargs)
        frame = df.backend_frame.map(map_func)
        return mpd.DataFrame(frame)

    @classmethod
    def has_multiindex(cls, input_dataframe, axis=0):
        """Function that is used to determine if dataframe/series has MultiIndex """
        if axis == 0:
            return isinstance(input_dataframe.index, pandas.MultiIndex)

        return isinstance(input_dataframe.columns, pandas.MultiIndex)

    @classmethod
    def transpose(cls, df, copy):
        """Compiling transpose"""
        map_func = ff.transpose(copy=copy)
        frame = df.backend_frame.map(map_func)
        frame.transpose()
        return mpd.DataFrame(frame)

    @classmethod
    def astype(cls, df, dtype, copy, errors):
        """Compiling astype"""
        func = ff.astype(dtype=dtype, copy=copy, errors=errors)

        if dtype == 'category':
            partition_shape = df.backend_frame.partition_shape
            frame = df.backend_frame.reduce(func, axis=0)
            frame = frame.repartition(partition_shape, i_config.get_min_block_size())
        else:
            frame = df.backend_frame.map(func)

        if isinstance(df, mpd.Series):
            return mpd.Series(frame)
        return mpd.DataFrame(frame)

    @classmethod
    def setitem(cls, input_dataframe, axis, key, value):
        """Compiling setitem"""
        # setitem for column
        if axis == 0:
            if is_scalar(value):
                set_item_fn = ff.set_item(key, value)
                input_dataframe.backend_frame = input_dataframe.backend_frame.map(set_item_fn)
                return

            set_item_fn = ff.set_item(key)
            if not isinstance(value, mpd.Series):
                value = cls.create_backend_frame(value, None, None, None, None)
                input_dataframe.backend_frame = input_dataframe.backend_frame.injective_map(value, None,
                                                                                            set_item_fn,
                                                                                            is_scalar=True,
                                                                                            need_repartition=True)
            else:
                input_dataframe.backend_frame = input_dataframe.backend_frame.injective_map(value.backend_frame,
                                                                                            None, set_item_fn,
                                                                                            is_scalar=True,
                                                                                            need_repartition=True)
            return

        # setitem to selected rows/cols along with axis
        setitem_part_func = ff.setitem_part_func(axis=axis, value=value)
        input_dataframe.backend_frame = input_dataframe.backend_frame.apply_select_indice(
            axis=axis,
            func=setitem_part_func,
            indices=None,
            labels=[key],
            new_index=input_dataframe.backend_frame.index,
            new_columns=input_dataframe.backend_frame.columns,
            keep_reminding=True)
        return

    @classmethod
    def setitem_elements(cls, input_dataframe, df_method, *args, **kwargs):
        """Compiling setitem specific for the case when a single element is updated"""
        func_name = getattr(df_method, "__name__", str(df_method))
        set_item_fn = ff.set_item_elements(func_name, **kwargs)
        input_dataframe.backend_frame = input_dataframe.backend_frame.setitem_elements(set_item_fn, *args)

    @classmethod
    def duplicated(cls, input_dataframe, subset=None, keep="first"):
        """Compiling duplicated"""
        if subset is not None:
            columns = input_dataframe.columns
            if not np.iterable(subset) or isinstance(subset, str) or isinstance(subset, tuple) and subset in columns:
                subset = (subset,)
            subset = cast(Iterable, subset)
            diff = pandas.Index(subset).difference(columns)
            if not diff.empty:
                raise KeyError(diff)
            if len(subset) == 1:
                df = input_dataframe[subset[0]]
            else:
                df = input_dataframe[list(subset)]
        else:
            df = input_dataframe
        if isinstance(df, mpd.DataFrame) and len(df.columns) > 1:
            hashed = df.apply(
                lambda s: hashlib.new("md5", str(tuple(s)).encode()).hexdigest(), axis=1
            )
        else:
            hashed = df
        func = ff.duplicated(keep=keep)
        duplicates = hashed.backend_frame.reduce(func, axis=0)
        return mpd.Series(duplicates)

    @classmethod
    def copy(cls, input_dataframe, deep):
        """Compiling copy"""
        frame = input_dataframe.backend_frame.copy(deep)
        return type(input_dataframe)(frame)

    @classmethod
    def series_comp_op(cls, input_series, func, other, scalar_other, level=None, fill_value=None, axis=0):
        """
        Compiling Series comparison operators.
        This function is not called from any external pandas API.
        """
        map_func = ff.series_comparison(func=func,
                                        level=level,
                                        fill_value=fill_value,
                                        axis=axis,
                                        is_scalar=scalar_other)
        if scalar_other:
            frame = input_series.backend_frame.injective_map(None, other, map_func, scalar_other)
        else:
            frame = input_series.backend_frame.injective_map_with_join(map_func, other.backend_frame)
        return mpd.Series(data=frame)

    @classmethod
    def df_comp_op(cls, input_dataframe, func, other, scalar_other, axis=0, level=None):
        """
        Compiling DataFrame comparison operators
        This function is not called from any external pandas API.
        """
        map_func = ff.df_comparison(func=func, level=level, axis=axis)
        if scalar_other:
            frame = input_dataframe.backend_frame.injective_map(None, other, map_func, scalar_other)
        else:
            frame = input_dataframe.backend_frame.injective_map_with_join(map_func, other.backend_frame)

        return mpd.DataFrame(data=frame)

    @classmethod
    def where(cls, df, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise', try_cast=None):
        """Compiling where"""
        df_row_split_points = df.backend_frame.get_axis_split_points(axis=0)
        df_col_split_points = df.backend_frame.get_axis_split_points(axis=1)
        cond_row_split_points = cond.backend_frame.get_axis_split_points(axis=0)
        cond_col_split_points = cond.backend_frame.get_axis_split_points(axis=1)
        if not np.array_equal(df_row_split_points, cond_row_split_points):
            cond.backend_frame.axis_repartition(axis=0, mblock_size=i_config.get_min_block_size(),
                                                by='split_pos', by_data=df_row_split_points)
        if not np.array_equal(df_col_split_points, cond_col_split_points):
            cond.backend_frame.axis_repartition(axis=1, mblock_size=i_config.get_min_block_size(),
                                                by='split_pos', by_data=df_col_split_points)
        func = ff.where(inplace=inplace,
                        axis=axis,
                        level=level,
                        errors=errors,
                        try_cast=try_cast)
        if is_scalar(other):
            frame = df.backend_frame.injective_map(cond.backend_frame, other, func, is_scalar=True)
        else:
            _, other = df.align(other, join='left')
            other_row_split_points = other.backend_frame.get_axis_split_points(axis=0)
            other_col_split_points = other.backend_frame.get_axis_split_points(axis=1)
            if not np.array_equal(df_row_split_points, other_row_split_points):
                other.backend_frame.axis_repartition(axis=0, mblock_size=i_config.get_min_block_size(),
                                                     by='split_pos', by_data=df_row_split_points)
            if not np.array_equal(df_col_split_points, other_col_split_points):
                other.backend_frame.axis_repartition(axis=1, mblock_size=i_config.get_min_block_size(),
                                                     by='split_pos', by_data=df_col_split_points)
            frame = df.backend_frame.injective_map(cond.backend_frame, other.backend_frame, func)
        if inplace is True:
            df.set_backend_frame(frame=frame)
            return None
        return mpd.DataFrame(frame)

    @classmethod
    def series_value_counts(cls, data, normalize, sort, ascending, bins, dropna):
        """Currently call default to pandas because it requires groupby to support (by is Series)."""
        return cls.default_to_pandas(data,
                                     data.value_counts,
                                     normalize=normalize,
                                     sort=sort,
                                     ascending=ascending,
                                     bins=bins,
                                     dropna=dropna,
                                     force_series=True)

    @classmethod
    def replace(cls, data, to_replace, value, inplace, limit, regex, method):
        """Compiling replace"""
        if isinstance(data, mpd.DataFrame):
            func = ff.replace(to_replace=to_replace,
                              value=value,
                              inplace=False,
                              limit=limit,
                              regex=regex,
                              method=method)
            if (is_scalar(to_replace) or isinstance(to_replace, (list, tuple))) and value is None:
                frame = data.backend_frame.reduce(func=func, axis=0)
            else:
                frame = data.backend_frame.map(func)
            if inplace is True:
                data.set_backend_frame(frame=frame)
                return None
            return mpd.DataFrame(frame)
        return None

    @classmethod
    def to_parallel(cls, input_dataframe):
        """Convert the DataFrame to Parallel class"""
        fn = ff.to_list()
        frame = input_dataframe.backend_frame.reduce(fn, axis=1)
        return mpd.Parallel(frame)

    @classmethod
    def reset_index(cls, input_dataframe, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        """Compiling reset_index"""
        frame = input_dataframe.backend_frame
        if not inplace:
            frame = frame.copy()

        new_index = pandas.RangeIndex(len(frame.index))
        if level is not None:
            if not isinstance(level, (tuple, list)):
                level = [level]
            level = [frame.index._get_level_number(lev) for lev in level]
            if len(level) < frame.index.nlevels:
                new_index = frame.index.droplevel(level)

        if not drop:
            # def get_level_name_and_values(frame, level, col_level, col_fill):
            if isinstance(frame.index, pandas.MultiIndex):
                names = [
                    (n if n is not None else f"level_{i}")
                    for i, n in enumerate(frame.index.names)
                ]
                to_insert = zip(frame.index.levels, frame.index.codes)
            else:
                default = "index" if "index" not in frame.columns else "level_0"
                names = [default] if frame.index.name is None else [frame.index.name]
                to_insert = ((frame.index, None),)

            multi_col = isinstance(frame.columns, pandas.MultiIndex)
            for i, (lev, lab) in reversed(list(enumerate(to_insert))):
                if not (level is None or i in level):
                    continue
                name = names[i]
                def process_name(name, col_fill):
                    col_name = list(name) if isinstance(name, tuple) else [name]
                    if col_fill is None:
                        if len(col_name) not in (1, frame.columns.nlevels):
                            raise ValueError(
                                "col_fill=None is incompatible "
                                f"with incomplete column name {name}"
                            )
                        col_fill = col_name[0]

                    lev_num = frame.columns._get_level_number(col_level)
                    name_lst = [col_fill] * lev_num + col_name
                    missing = frame.columns.nlevels - len(name_lst)
                    name_lst += [col_fill] * missing
                    name = tuple(name_lst)
                    return name
                if multi_col:
                    name = process_name(name, col_fill)
                # to ndarray and maybe infer different dtype
                level_values = lev.values
                if level_values.dtype == np.object_:
                    level_values = maybe_convert_objects(level_values)

                if lab is not None:
                    # if we have the codes, extract the values with a mask
                    level_values = algorithms.take(
                        level_values, lab, allow_fill=True, fill_value=lev._na_value
                    )
                    # return name, level_values

                # name, level_values = get_level_name_and_values(frame, level, col_level, col_fill)
                part_loc, internal_loc = frame.get_internal_loc(0, axis=1, allow_append=True)
                insert_func = ff.insert(loc=internal_loc, column=name, value=level_values)
                concat_func = ff.concat()
                frame = frame.apply_select_indice_axis(insert_func, concat_func, indice=part_loc, axis=0)
            frame = frame.set_index(new_index)
            return mpd.DataFrame(data=frame)

        frame.index = new_index
        return mpd.DataFrame(data=frame)

    @classmethod
    def set_axis(cls, input_dataframe, labels, axis=0, inplace=False):
        """Compiling set_axis"""
        if isinstance(input_dataframe, mpd.DataFrame):
            frame = input_dataframe.backend_frame
            if not inplace:
                frame = frame.copy()
            # change index
            if axis in (0, 'index'):
                frame = frame.set_index(labels)
                return mpd.DataFrame(data=frame)
            # change columns
            # If new labels are as same as current columns, skip to_pandas() to reduce time wasting.
            if frame.columns.equals(labels):
                return input_dataframe

            # added column_setter in EagerFrame, can assign value to columns directly
            frame.columns = labels
            if inplace:
                return mpd.DataFrame(data=frame, columns=labels)
            return mpd.DataFrame(data=frame)

        if isinstance(input_dataframe, mpd.Series):
            series = input_dataframe.backend_frame
            if not inplace:
                series = series.copy()
            series = series.set_index(labels)
            return mpd.Series(data=series)

        return None

    @classmethod
    def append_column(cls, input_dataframe, key, value):
        """
        Function of appending column.
        This function is not called from any external pandas API.
        """
        if isinstance(value, (mpd.DataFrame, mpd.Series)):
            if isinstance(value, mpd.Series):
                value = value.to_frame(name=key)
            frame = input_dataframe.backend_frame.append_column(value.backend_frame)
            return mpd.DataFrame(frame)
        key_value = {key: value}
        appending_frame = cls.create_backend_frame(key_value, None, None, None, None)
        frame = input_dataframe.backend_frame.append_column(appending_frame)
        return mpd.DataFrame(frame)

    @classmethod
    def invert(cls, input_dataframe):
        """Compiling invert"""
        fn = ff.invert()
        frame = input_dataframe.backend_frame.map(fn)
        return mpd.DataFrame(data=frame)

    @classmethod
    def set_series_name(cls, input_dataframe, value):
        """Compiling set series name function"""
        fn = ff.set_series_name(value=value)
        input_dataframe.backend_frame = input_dataframe.backend_frame.map(fn)

    @classmethod
    def to_frame(cls, ser, name):
        """Compiling to_frame"""
        hash(name)
        self_cp = ser.copy()

        def ser_to_df(df):
            if name is not None:
                df.columns = [name]
            if name is None and df.columns == '__unsqueeze_series__':
                df.columns = pandas.RangeIndex(start=0, stop=1, step=1)
            return df

        frame = self_cp.backend_frame.map(ser_to_df)
        return mpd.DataFrame(frame)

    @classmethod
    def to_csv(cls, df, path_or_buf=None, **kwargs):
        """Compiling to_csv"""
        def row_to_csv(row):
            output = row.to_csv(**kwargs)
            return output

        columns = kwargs.get('columns')
        index_label = kwargs.get('index_label')
        mpd_types = (mpd.DataFrame, mpd.Series)
        if isinstance(columns, mpd_types):
            columns = kwargs.update({"columns": columns.to_pandas()})
        if isinstance(index_label, mpd_types):
            index_label = kwargs.update({"index_label": index_label.to_pandas()})

        if kwargs.get('header'):
            head_frame = df.head(0).backend_frame.reduce(row_to_csv, axis=1)
            csv_header = mpd.DataFrame(head_frame)[0][0]
        else:
            csv_header = None

        kwargs['header'] = False

        frame = df.backend_frame
        csv_rows = mpd.DataFrame(frame.reduce(row_to_csv, axis=1))
        csv_body = "".join(csv_rows[0])

        if csv_header is None:
            csv_str = csv_body
        else:
            csv_str = csv_header + csv_body

        if path_or_buf is None:
            return csv_str

        cls.write_csv_str(path_or_buf, csv_str, **kwargs)

    @classmethod
    def write_csv_str(cls, path_or_buf, csv_str, mode, encoding, errors,
                      compression, storage_options, **kwargs):
        """Function that handling write csv strings"""
        with get_handle(
                path_or_buf,
                mode=mode,
                encoding=encoding,
                errors=errors,
                compression=compression,
                storage_options=storage_options,
        ) as handles:
            handles.handle.write(csv_str)

    @classmethod
    def remote_to_numpy(cls, df):
        def func(data):
            return data.to_numpy()
        frame = df.backend_frame.map(func)
        parts = frame.partitions
        return parts

    @classmethod
    def memory_usage(cls, input_dataframe, index=True, deep=False):
        """Compiling memory usage"""
        map_func = ff.memory_usage(index=index, deep=deep)
        return mpd.Series(input_dataframe.backend_frame.reduce(map_func, axis=0))


def get_groupby_reduce_functions(groupby_method_name, by_names, **kwargs):
    """
    The function mapping dictionary of the groupby map and reduce functions .
    """
    groupby_reduce_functions = {
        "all": (ff.groupby_map('all', **kwargs),
                ff.groupby_reduce('all', by_names, **kwargs)),
        "any": (ff.groupby_map('any', **kwargs),
                ff.groupby_reduce('any', by_names, **kwargs)),
        "count": (ff.groupby_map('count', **kwargs),
                  ff.groupby_reduce('sum', by_names, **kwargs)),
        "max": (ff.groupby_map('max', **kwargs),
                ff.groupby_reduce('max', by_names, **kwargs)),
        "min": (ff.groupby_map('min', **kwargs),
                ff.groupby_reduce('min', by_names, **kwargs)),
        "prod": (ff.groupby_map('prod', **kwargs),
                 ff.groupby_reduce('prod', by_names, **kwargs)),
        "size": (ff.groupby_map('size', **kwargs),
                 ff.groupby_reduce('sum', by_names, **kwargs)),
        "sum": (ff.groupby_map('sum', **kwargs),
                ff.groupby_reduce('sum', by_names, **kwargs)),
    }
    output = groupby_reduce_functions[groupby_method_name]
    return output


def get_groupby_reduce_dict_functions(agg_func, by_names, **kwargs):
    """
    Group underlying data and apply agg functions to each group of the specified column/row.
    """
    map_dict = {}
    reduce_dict = {}
    for col, col_funcs in agg_func.items():
        map_dict[col], reduce_dict[col] = get_groupby_reduce_functions(col_funcs, by_names, **kwargs)
    return map_dict, reduce_dict
