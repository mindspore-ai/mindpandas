# Copyright 2023 Huawei Technologies Co., Ltd
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
This module is used to build logical plan for lazy mode
"""
import pandas
from pandas._libs.lib import no_default, is_scalar
import mindpandas as mpd
from mindpandas.compiler.function_factory import FunctionFactory as ff
from .query_plan import Function, fn_op_map, Operator
from .statistics import Statistics
from .workspace import WorkSpace


class LogicalPlanBuilder:
    """
    Used to build logical plan for lazy mode
    """
    dag = WorkSpace.dag

    @classmethod
    def cross_reference(cls, node_id, pd_object, materialized=False):
        pd_object.node_id = node_id
        if materialized:
            cls.dag.cache(node_id, pd_object)

    @classmethod
    def leaf_dataframe(cls, df, index, columns):
        stats = Statistics(len(index), len(columns), shape=(16, 16))
        kwargs: dict = {'index': index, 'columns': columns}
        node_id = cls.dag.add_source_node(name=fn_op_map[Function.DATAFRAME], fn=Function.DATAFRAME,
                                          stats=stats, **kwargs)
        # Attach the materialized DataFrame object to the node
        cls.cross_reference(node_id, df, materialized=True)
        return df

    @classmethod
    def leaf_series(cls, s, index):
        stats = Statistics(len(index), shape=(16, 1))
        kwargs: dict = {'index': index}
        node_id = cls.dag.add_source_node(name=fn_op_map[Function.SERIES], fn=Function.SERIES,
                                          stats=stats, **kwargs)
        # Attach the materialized Series object to the node
        cls.cross_reference(node_id, s, materialized=True)
        return s

    @classmethod
    def read_csv(cls, filepath, **kwargs):
        """
        Build a logical operator to represent the read_csv()
        """
        kwargs['filepath'] = filepath
        chunksize = kwargs.pop('chunksize')
        iterator = kwargs.pop('iterator')
        if chunksize or iterator:
            if chunksize is None:
                chunksize = 1
            fn = ff.csv_iterator(filepath, chunksize, **kwargs)
            kwargs: dict = {'func': fn}
            t = Function.UDF
            node_id = cls.dag.add_source_node(
                name=Operator.SOURCE, fn=t, **kwargs)
        else:
            node_id = cls.dag.add_source_node(
                name=fn_op_map[Function.READ_CSV], fn=Function.READ_CSV, **kwargs)
        df = mpd.DataFrame()
        cls.cross_reference(node_id, df)

        # add aux node
        # kwargs: dict = {'x': 1}

        # aux_node_id = cls.dag.add_1op_node(name=Operator.AUX, func=Function.AUX,
        #                                child=df.node_id, **kwargs)
        # aux_df = mpd.DataFrame()
        # cls.cross_reference(aux_node_id, aux_df)

        return df

    @classmethod
    def groupby(
            cls, input_dataframe,
            by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True,
            squeeze: bool = no_default, observed=False, dropna=True
    ):
        """
        Build a logical operator to represent the groupby()
        """
        kwargs: dict = {'by': by, 'axis': axis, 'level': level, 'as_index': as_index, 'sort': sort,
                        'group_keys': group_keys, 'squeeze': squeeze, 'observed': observed, 'dropna': dropna}
        node_id = cls.dag.add_1op_node(name=fn_op_map[Function.GROUPBY], fn=Function.GROUPBY,
                                       child=input_dataframe.node_id, **kwargs)

        from ..groupby import DataFrameGroupBy

        df = DataFrameGroupBy(input_dataframe)
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def groupby_default_to_pandas(cls, input_dataframe, groupby_method_name, **kwargs):
        """
        The purpose of this function is to mimic the API of the same function in class QueryCompiler
        in order to give a meaningful error message in groupby_reduce().
        For example if we call `df.groupby(by='c1').mean()`, the code will go through `DataFrameGroupBy.mean()`.
        In lazy mode, the variable _qc in DataFrameGroupBy maps to this class. We then depend on the code
        in `groupby_reduce()` to flag out the aggregate functions that do not implement in mindspore.pandas.
        """
        output_df = cls.groupby_reduce(groupby_method_name, input_dataframe, drop=None, by=None, axis=None,
                                       by_dataframe=None, by_names=None, **kwargs)
        return output_df

    @classmethod
    def math_op(cls, input_dataframe, op, other, axis='columns', level=None, fill_value=None, **kwargs):
        """
        Build a logical operator to represent the math_op
        """
        kwargs['op'] = op
        kwargs['axis'] = axis
        kwargs['level'] = level
        kwargs['fill_value'] = fill_value
        if is_scalar(other):
            kwargs['other'] = other
            node_id = cls.dag.add_1op_node(name=fn_op_map[Function.MATH], fn=Function.MATH,
                                           child=input_dataframe.node_id, **kwargs)
        else:
            if isinstance(other, pandas.DataFrame):
                other = mpd.DataFrame(data=other)
                print(other)
                print(other.node_id)
            # possible to add df to itself - create new object in that case
            if input_dataframe.node_id == other.node_id:
                other = mpd.DataFrame(other)
            node_id = cls.dag.add_2op_node(name=fn_op_map[Function.MATH], fn=Function.MATH,
                                           left_child=input_dataframe.node_id, right_child=other.node_id, **kwargs)
        df = mpd.DataFrame()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def stat_op(cls, input_dataframe, op_name, **kwargs):
        """
        Build a logical operator to represent the stat_op
        """
        stat_functions = {
            "mean": {"name": fn_op_map[Function.MEAN], "fn": Function.MEAN},
            "min": {"name": fn_op_map[Function.MIN], "fn": Function.MIN},
            "max": {"name": fn_op_map[Function.MAX], "fn": Function.MAX},
            "median": {"name": fn_op_map[Function.MEDIAN], "fn": Function.MEDIAN},
            "count": {"name": fn_op_map[Function.COUNT], "fn": Function.COUNT},
            "sum": {"name": fn_op_map[Function.SUM], "fn": Function.SUM}
        }
        if op_name not in stat_functions:
            raise NotImplementedError("Operation not supported")

        functions = stat_functions[op_name]
        node_id = cls.dag.add_1op_node(name=functions["name"], fn=functions["fn"],
                                       child=input_dataframe.node_id, **kwargs)
        df = mpd.Series()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def groupby_reduce(cls, op_name, input_dataframe, **kwargs):
        """
        Build a logical operator to represent the groupby_reduce()
        """
        # TODO: change to Dictionary
        if op_name == 'count':
            func = Function.COUNT
        elif op_name == 'sum':
            func = Function.SUM
        elif op_name == 'min':
            func = Function.MIN
        elif op_name == 'max':
            func = Function.MAX
        elif op_name == 'size':
            func = Function.SIZE
        # Add these functions in when we can run them natively. Currently we redirect to pandas.
        # elif op_name == 'mean':
        #     func = Function.MEAN
        # elif op_name == 'median':
        #     func = Function.MEDIAN
        else:
            raise NotImplementedError(
                f"Aggregate function '{op_name}' is not yet supported in lazy mode.")

        node_id = cls.dag.add_1op_node(
            name=Operator.REDUCEBYKEY, fn=func, child=input_dataframe.node_id, **kwargs)
        if isinstance(input_dataframe, mpd.groupby.SeriesGroupBy):
            df = mpd.Series()
        else:
            df = mpd.DataFrame()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def merge(cls, **kwargs):
        left = kwargs.pop('left')
        right = kwargs.pop('right')
        operands = [left.node_id, right.node_id]
        node_id = cls.dag.add_op_node(name=fn_op_map[Function.MERGE], fn=Function.MERGE,
                                      children=operands, **kwargs)
        df = mpd.DataFrame()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def default_to_pandas(cls, df, df_method, *args, force_series=False, **kwargs):
        if isinstance(type(df_method), str):
            # some ops have sub methods such as MathOp -> {add, sub, mul, div}
            print("in logical_plan ", df_method)
            raise NotImplementedError(
                f"The function `{df_method} has not yet implemented in lazy mode.")
        raise NotImplementedError(
            f"The function `{df_method.__name__} has not yet implemented in lazy mode.")

    @classmethod
    def series_comp_op(cls, input_obj, op, other, scalar_other, level=None, fill_value=None, axis=0):
        """
        Comparison operators, i.e., ==, <, <=, >, >=, and !=
        :param input_obj: Series
        :param op: the input comparison operator
        :param other: Series or DataFrame or scalar value (currently only value is supported.)
        :param level: only None is supported
        :param fill_value: only None is supported
        :param axis: 0 or index (only index axis is supported)
        :return: Series of Boolean type of the same shape as the input
        """
        if not isinstance(input_obj, mpd.Series):
            raise NotImplementedError(
                f"object {type(input_obj).__name__} is not supported yet.")
        # Other semantic checks are done before calling this function.

        # Build a logical operator to represent the comparison operator
        kwargs: dict = {'op': op, 'other': other, 'axis': axis, 'scalar_other': scalar_other, 'level': level,
                        'fill_value': fill_value}
        node_id = cls.dag.add_1op_node(name=fn_op_map[Function.COMPOP], fn=Function.COMPOP,
                                       child=input_obj.node_id, **kwargs)
        if isinstance(input_obj, mpd.Series):
            new_obj = mpd.Series()
        elif isinstance(input_obj, mpd.DataFrame):
            new_obj = mpd.DataFrame()
        else:
            raise RuntimeError(f"invalid input {type(input_obj).__name__}.")

        cls.cross_reference(node_id, new_obj)
        return new_obj

    @classmethod
    def setitem(cls, input_obj, axis, key, other):
        """
        Build a logical operator to represent the setitem
        """
        # check the difference between this function and setitem_1() in this file, combine them later
        if not isinstance(other, (mpd.Series, mpd.DataFrame)):
            raise NotImplementedError(
                f"object {type(input_obj).__name__} is not supported yet.")
        # Other semantic checks are done before calling this function.
        # Build a logical operator to represent the comparison operator
        kwargs: dict = {'axis': axis, 'columns': key}
        node_id = cls.dag.add_2op_node(name=fn_op_map[Function.SETITEM], fn=Function.SETITEM,
                                       left_child=input_obj.node_id, right_child=other.node_id, **kwargs)
        input_obj.node_id = node_id

    @classmethod
    def getitem_array(cls, input_obj, key):
        """

        :param input_obj: input Series or DataFrame
        :param key: can be one of the following forms:
            - an array of column(s), which is equivalent to ???
            - a Series of Boolean values, which will be used to filter rows
            - an expression that evaluated to a Series of Boolean values
        :return: a subset of the input Series or DataFrame
        """
        if isinstance(key, mpd.Series):
            node_id = cls._select(input_obj, key)
        elif isinstance(key, slice):
            node_id = cls._view(input_obj, key)
        else:
            node_id = cls._project_n_columns(input_obj, key)

        if isinstance(input_obj, mpd.Series):
            new_obj = mpd.Series()
        elif isinstance(input_obj, mpd.DataFrame):
            new_obj = mpd.DataFrame()
        cls.cross_reference(node_id, new_obj)
        return new_obj

    @classmethod
    def setitem_1(cls, input_obj, axis, key, value):
        """
        updates the input dataframe along axis and key
        with the given value.

        :param input_obj: input DataFrame
        :param key: can be one of the following forms:
            - an array of column(s)
        :param value: a DataFrame
        :return: updates a subset of DataFrame
        """
        if not isinstance(input_obj, mpd.Series):
            # TODO: use map instead of setitem
            # node_id = cls.dag.add_2op_node(name=Operator.SETITEM, func=Function.SETITEM,
            #                            left_child=input_obj.node_id, right_child=value.node_id, **kwargs)
            kwargs: dict = {}
            node_id_aux = cls.dag.add_1op_node(name=Operator.AUX, fn=Function.AUX,
                                               child=input_obj.node_id, **kwargs)
            df = mpd.DataFrame()
            cls.cross_reference(node_id_aux, df)
            kwargs: dict = {'axis': axis, 'columns': key}
            node_id = cls.dag.add_2op_node(name=fn_op_map[Function.SETITEM], fn=Function.SETITEM,
                                           left_child=df.node_id, right_child=value.node_id, **kwargs)
            input_obj.node_id = node_id
        else:
            raise NotImplementedError

    @classmethod
    def _select(cls, input_obj, predicates):
        """
        select column(s) from input object.
        """
        operands = [input_obj.node_id]
        if isinstance(predicates, list):
            for p in predicates:
                operands.append(p.node_id)
        else:
            operands.append(predicates.node_id)
        node_id = cls.dag.add_op_node(name=fn_op_map[Function.SELECT], fn=Function.SELECT,
                                      children=operands)
        return node_id

    @classmethod
    def _project_n_columns(cls, input_obj, columns):
        """
        Return a list of column(s) from the input DataFrame, e.g., df[['col1',...]]
        :param input_obj: input DataFrame
        :param column: list of column name(s) of the input DataFrame
        :return: a DataFrame of the projected column(s)
        """
        # columns is a list of str representing column names
        if not isinstance(columns, list) or any(not isinstance(col, str) for col in columns):
            raise RuntimeError("A list of str is expected.")
        kwargs: dict = {'columns': columns}
        node_id = cls.dag.add_1op_node(name=fn_op_map[Function.PROJECTN], fn=Function.PROJECTN,
                                       child=input_obj.node_id, **kwargs)
        return node_id

    @classmethod
    def _project_1_column(cls, input_obj, column):
        """
        project single column
        """
        # column is a str representing a column name
        if not isinstance(column, str):
            # This must be a KeyError exception because in the caller DataFrame.__getitem__,
            # it could be a legitimate error. The caller will catch the KeyError  and proceed further.
            raise KeyError("A str is expected.")
        kwargs: dict = {'column': column}
        node_id = cls.dag.add_1op_node(name=fn_op_map[Function.PROJECT1], fn=Function.PROJECT1,
                                       child=input_obj.node_id, **kwargs)
        return node_id

    @classmethod
    def _view(cls, input_obj, key):
        # This is only for supporting head, tail, and getitem with index (not fully support loc/iloc)
        kwargs: dict = {'key': key}
        node_id = cls.dag.add_1op_node(name=fn_op_map[Function.VIEW], fn=Function.VIEW,
                                       child=input_obj.node_id, **kwargs)
        return node_id

    @classmethod
    def getitem_column(cls, input_obj, column):
        """
        Return one column from the input DataFrame, e.g., df.col1 or df['col1']
        :param input_obj: input DataFrame
        :param column: can be one of the following forms:
            - a str representing a column name of the input DataFrame
            - a sub-plan returning a Series of Boolean type
        :return: a Series of the projected column
        """
        if isinstance(column, mpd.Series):
            node_id = cls._select(input_obj, column)
            new_obj = mpd.DataFrame()
        else:
            # When project out one column e.g., df['col1'], the output is converted to a Series.
            # Note df[['col1']] has a slightly different semantics, the output remains a DataFrame.
            node_id = cls._project_1_column(input_obj, column)
            new_obj = mpd.Series()

        cls.cross_reference(node_id, new_obj)
        return new_obj

    @classmethod
    def map(cls, input_obj, fn):
        kwargs: dict = {'func': fn}
        t = Function.UDF
        node_id = cls.dag.add_1op_node(name=Operator.MAP1, fn=t,
                                       child=input_obj.node_id, **kwargs)
        df = type(input_obj)()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def reduce(cls, input_obj, fn):
        kwargs: dict = {'func': fn}
        t = Function.UDF
        node_id = cls.dag.add_1op_node(name=Operator.REDUCE, fn=t,
                                       child=input_obj.node_id, **kwargs)
        df = type(input_obj)()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def stream_source(cls, fn, is_row=True):
        kwargs: dict = {'func': fn, 'is_row': is_row}
        t = Function.UDF
        node_id = cls.dag.add_source_node(name=Operator.SOURCE, fn=t, **kwargs)
        df = mpd.Parallel()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def reducebykey(cls, input_obj, keyby_fn, reduce_fn):
        kwargs: dict = {'keyby_fn': keyby_fn, 'reduce_fn': reduce_fn}
        t = Function.UDF
        node_id = cls.dag.add_1op_node(name=Operator.REDUCEBYKEY, fn=t,
                                       child=input_obj.node_id, **kwargs)
        df = type(input_obj)()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def file_sink(cls, input_obj, filepath, chunksize, timestamp_in_name=False):
        kwargs: dict = {'filepath': filepath, 'chunksize': chunksize,
                        'timestamp_in_name': timestamp_in_name}
        t = Function.TO_CSV
        node_id = cls.dag.add_1op_node(name=Operator.SINK, fn=t,
                                       child=input_obj.node_id, **kwargs)
        df = type(input_obj)()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def _create_node(cls, input_obj, fn=None, **kwargs):

        node_id = cls.dag.add_1op_node(fn_op_map[fn], fn,
                                       child=input_obj.node_id, **kwargs)
        df = type(input_obj)()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def apply(cls, input_obj, **kwargs):
        return cls._create_node(input_obj, Function.APPLY, **kwargs)

    @classmethod
    def replace(cls, input_obj, **kwargs):
        return cls._create_node(input_obj, Function.REPLACE, **kwargs)

    @classmethod
    def fillna(cls, **kwargs):
        input_obj = kwargs.pop("input_dataframe", None)
        return cls._create_node(input_obj, Function.FILLNA, **kwargs)

    @classmethod
    def map_op(cls, input_dataframe, op_name, **kwargs):
        """
        Build a logical operator to represent the map_op
        Can be generalized to more pure map operations
        """
        stat_functions = {
            "applymap": {"name": fn_op_map[Function.APPLYMAP], "fn": Function.APPLYMAP}}

        if op_name not in stat_functions:
            raise NotImplementedError("Operation not supported")

        functions = stat_functions[op_name]
        node_id = cls.dag.add_1op_node(name=functions["name"], fn=functions["fn"],
                                       child=input_dataframe.node_id, **kwargs)
        df = type(input_dataframe)()
        cls.cross_reference(node_id, df)
        return df

    @classmethod
    def drop(cls, input_obj, **kwargs):
        return cls._create_node(input_obj, Function.DROP, **kwargs)
