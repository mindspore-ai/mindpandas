# Copyright 2020 Huawei Technologies Co., Ltd
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
# ==============================================================================

"""
Module contains ``Util`` class which is responsible for supporting functions and
functionality for ``test_pandas.py``.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from pandas._libs.lib import is_list_like
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
from pandas._testing import assert_dict_equal
import pytest

import mindpandas as mpd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TestUtil:
    def __init__(self):
        self.filename = os.path.join(DATA_DIR, "test_simple.csv")
        self.default_create_fn = self.create_df_range
        self.default_compare_fn = self.run_compare
        self.rows = 100
        self.columns = 4
        self.unique = 5
        self.use_csv = False
        self.perf_materialize = False
        self.partition_row_size = 16
        self.partition_column_size = 16

    def compare(self, function, create_fn=None, prep_fn=None, **kwargs):
        """
        This is main function the user should call to test their function.
        """
        if kwargs.pop("skip", False):
            warnings.warn(f"test skipped: function = {function.__name__}, create_fn = {create_fn.__name__}")
        else:
            self.default_compare_fn(function, create_fn, prep_fn, **kwargs)

    def set_perf_mode(self, materialize=False, partition_row_size=16, partition_column_size=16):
        """
        High level options to set. Set the perf mode.
        """
        self.perf_materialize = materialize
        self.partition_row_size = partition_row_size
        self.partition_column_size = partition_column_size
        self.default_compare_fn = self.run_all_compare_time

    def set_use_csv(self):
        """
        Set attribute use_csv.
        """
        self.use_csv = True
        self.init_csv_cache()

    def unset_use_csv(self):
        """
        Unset attribute use_csv.
        """
        self.use_csv = False

    def set_size(self, rows=None, cols=None, unique=None):
        """
        Set attributes rows, cols and unique.
        """
        if rows is not None:
            self.rows = rows
        if cols is not None:
            self.columns = cols
        if unique is not None:
            self.unique = unique
        print("setting rows ", self.rows, " columns ",
              self.columns, " unique ", self.unique)
        self.init_csv_cache()

    def set_filename(self, filename):
        """
        Set attribute filename.
        """
        self.filename = filename

    def set_default_create_fn(self, create_fn):
        """
        Set attribute default_create_fn.
        """
        self.default_create_fn = create_fn

    def init_csv_cache(self):
        """
        Initialize csv cache.
        """
        if self.use_csv:
            print("Creating temp csv files, this make take some time")
            module = pd
            self.create_df_range(module, create_cache=True)
            self.create_df_range_float(module, create_cache=True)
            self.create_single_column_df(module, create_cache=True)
            self.create_df_gb_frame(module, create_cache=True)
            print("Done creating temp csv files, this make take some time")

    def noop(self, df):
        """
        Return a built-in empty fn.
        """
        return df

    # built-in create functions that can be used
    def create_df_readcsv(self, module):
        """
        Return a DataFrame from readcsv.
        """
        df = module.read_csv(self.filename)
        print("in create df readcsv", df)
        return df

    def create_df_array(self, module):
        """
        Return a DataFrame that is from numpy.array.
        """
        min_val = 0
        max_val = self.unique
        np.random.seed(100)
        data = np.random.randint(
            min_val, max_val, size=(self.rows, self.columns))
        df = module.DataFrame(data)
        return df

    def create_df_array_with_nan(self, module):
        np.random.seed(100)
        arr = np.concatenate((
            np.random.randint(1000, size=int(self.rows * self.columns * (1 - 0.3))),
            np.full(shape=int(self.rows * self.columns * 0.3), fill_value=np.nan),), axis=0)
        np.random.shuffle(arr)
        return module.DataFrame(arr.reshape(self.rows, self.columns), columns=[str(c) for c in range(self.columns)])

    def create_df_bool(self, module):
        """
        Return a boolean DataFrame that is from numpy.array.
        """
        np.random.seed(100)
        bools = np.random.randint(2, size=(self.rows, self.columns))
        bools[0] = np.zeros(self.columns)
        bools[-1] = np.ones(self.columns)
        df = module.DataFrame(bools.T)
        return df

    def create_df_bool_and_str(self, module):
        """
        Return a DataFrame that values have string and bool.
        """
        df = module.DataFrame({"A": ["A", "A", True, False],
                               "B": ["B", "B", "B", "B"],
                               "C": [True, "C", False, "C"],
                               "D": [True, True, True, True]})
        return df

    def create_df_int_and_str(self, module):
        """
        Return a DataFrame that values have string and int.
        """
        df = module.DataFrame({"A": ["1", "2", 3, 4],
                               "B": ["1", "2", "3", "4"],
                               "C": [1, "2", 3, "4"],
                               "D": [1, 2, 3, 4]})
        return df

    def create_df_gaussian(self, module, mean=0, var=1):
        """
        Return a DataFrame that values are from gussian sampling.
        """
        np.random.seed(100)
        guassian_sampling = mean + var * \
                            np.random.randn(self.rows, self.columns)
        df = module.DataFrame(guassian_sampling)
        return df

    def create_df_large_cols(self, module):
        """
        Return a DataFrame that has large columns.
        """
        np.random.seed(100)
        sampling = np.random.randint(1, 10, size=(4, 100))
        df = module.DataFrame(sampling)
        return df

    def create_df_empty(self, module):
        """
        Return an empty DataFrame.
        """
        return module.DataFrame()

    def create_df_empty_with_columns(self, module):
        """
        Return an empty DataFrame that has columns input.
        """
        return module.DataFrame(data=None, columns=[1, 2, 3, 4])

    def create_df_range(self, module, create_cache=False):
        """
        Return a specific type DataFrame.
        """
        name = os.path.join(DATA_DIR, "create_df_range.csv")
        if self.use_csv and not create_cache:
            df = module.read_csv(name)
        else:
            data = np.arange(self.rows * self.columns, 0, -1).reshape(self.rows, self.columns)
            data = data * 3
            # create duplicate row
            data[1] = data[3]
            columns = [chr(i) for i in range(65, 65 + self.columns)]
            df = module.DataFrame(data, columns=columns)
            if create_cache:
                df.to_csv(name)
        return df

    def create_df_index_with_nan(self, module):
        np.random.seed(100)
        data = np.random.randn(5, 10)
        return module.DataFrame(data=data, index=[3, 2, np.nan, 1, np.nan])

    def create_df_index_str_list(self, module):
        """
        Return a specific type DataFrame.
        """

        data = np.arange(self.rows * self.columns, 0, -1).reshape(self.rows, self.columns)
        data = data * 3
        # create duplicate row
        data[1] = data[3]
        columns = [chr(i) for i in range(65, 65 + self.columns)]
        index = [str(i) for i in range(self.rows)]

        df = module.DataFrame(data, index=index, columns=columns)
        return df

    def create_df_index_integer_list(self, module):
        """
        Return a specific type DataFrame.
        """
        data = np.arange(self.rows * self.columns, 0, -1).reshape(self.rows, self.columns)
        data = data * 3
        # create duplicate row
        data[1] = data[3]
        columns = [chr(i) for i in range(65, 65 + self.columns)]
        index = [i for i in range(self.rows)]
        df = module.DataFrame(data, index=index, columns=columns)
        return df

    def create_df_index_range(self, module):
        """
        Return a specific type DataFrame.
        """
        data = np.arange(self.rows * self.columns, 0, -1).reshape(self.rows, self.columns)
        data = data * 3
        # create duplicate row
        data[1] = data[3]
        columns = [chr(i) for i in range(65, 65 + self.columns)]
        index = pd.RangeIndex(start=0, stop=self.rows, step=1)

        df = module.DataFrame(data, index=index, columns=columns)
        return df

    def create_df_index_str_list_2(self, df, offset):
        """
        Return a specific type DataFrame.
        """
        rows = self.rows
        columns = self.columns
        data = np.arange(rows * columns, 0, -1).reshape(rows, columns)
        index = [str(i + offset) for i in range(len(df.index))]
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(data, index=index)
        return mpd.DataFrame(data, index=index)

    def create_df_index_integer_list_2(self, df, offset):
        """
        Return a specific type DataFrame.
        """
        rows = self.rows
        columns = self.columns
        data = np.arange(rows * columns, 0, -1).reshape(rows, columns)
        index = [i + offset for i in range(len(df.index))]
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(data, index=index)
        return mpd.DataFrame(data, index=index)

    def create_df_index_range_2(self, df, offset):
        """
        Return a specific type DataFrame.
        """
        rows = self.rows
        columns = self.columns
        data = np.arange(rows * columns, 0, -1).reshape(rows, columns)
        index = pd.RangeIndex(start=0 - offset, stop=rows - offset, step=1)
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(data, index=index)
        return mpd.DataFrame(data, index=index)

    def create_df_has_duplicate_index(self, module, seed=100):
        """
        Return a DataFrame that has duplicated index.
        """
        np.random.seed(seed)
        data = np.random.randint(1000, size=(self.rows, self.columns))
        index = np.random.randint(self.rows // 2, size=self.rows)
        columns = [chr(i) for i in range(65, 65 + self.columns)]
        df = module.DataFrame(data, index=index, columns=columns)
        return df

    def create_df_mixed_dtypes(self, module):
        """
        Return a DataFrame that has mixed dtypes.
        """
        data = {'City': ['Beijing', 'Jinan', 'Tianjin',
                         'Shanghai', 'Hangzhou', 'Chengdou',
                         'Aomen', 'Nanjing'],
                'AverageIncome': [10000, np.nan, 8000, 15000,
                                  12000, 7000, 10000, 9000],
                'Popularity': [50000, 40000, 53000, 20000,
                               25000, 10000, 15000, 30000]}
        return module.DataFrame(data)

    def create_df_mixed_dtypes_2(self, module):
        """
        Return a DataFrame that has mixed dtypes.
        """
        data = {'City': ['Beijing', np.nan, 'Tianjin',
                         'Shanghai', 'Hangzhou', 'Chengdou',
                         'Aomen', 'Nanjing'],
                'AverageIncome': ['string', np.nan, np.nan, np.nan,
                                  12000, 7000, 10000, 9000],
                'Popularity': [50000, 40000, 53000, 20000,
                               25000, 10000, 15000, 30000]}
        return module.DataFrame(data)

    def create_df_range_float(self, module, create_cache=False):
        """
        Return a specific type DataFrame.
        """
        name = os.path.join(DATA_DIR, "create_df_range_float.csv")
        if self.use_csv and not create_cache:
            df = module.read_csv(name)
        else:
            data = np.arange(self.rows * self.columns, 0, -1,
                             np.float64).reshape(self.rows, self.columns)
            data = data * 3
            # create duplicate row and nan
            data[1] = data[3]
            data[4][1] = np.nan
            data[5][3] = np.nan
            columns = [chr(i) for i in range(65, 65 + self.columns)]
            df = module.DataFrame(data, columns=columns)
            if create_cache:
                df.to_csv(name)
        return df

    def create_df_merge(self, module):
        '''
        import random
        random.seed(seed)
        index = [f"row{i}" for i in range(50)] * 3
        lkey = random.sample(index, k=rows)
        rkey = random.sample(index, k=rows)
        pdf1 = pd.DataFrame({'lkey': lkey,
                             'value': range(rows)})
        pdf2 = pd.DataFrame({'rkey': rkey,
                             'value': range(rows)})
        '''
        # have keys in different order in each table
        lkey = np.arange(0, self.rows, 1)
        rkey = np.arange(self.rows, 0, -1)
        # create a couple of duplicate keys to check functionality
        lkey[1] = lkey[5]
        rkey[1] = rkey[4]
        pdf1 = pd.DataFrame({'lkey': lkey,
                             'value': range(self.rows)})
        pdf2 = pd.DataFrame({'rkey': rkey,
                             'value': range(self.rows)})
        df = (module.DataFrame(pdf1), module.DataFrame(pdf2))
        return df

    def create_df_small(self, module):
        """
        Return a small size DataFrame with columns.
        """
        df = module.DataFrame([[1, 2, 3, 4],
                               [2, 4, 6, 8],
                               [3, 6, 9, 12],
                               [4, 8, 12, 16],
                               [5, 10, 15, 20],
                               [6, 12, 18, 24]],
                              columns=['A', 'B', 'C', 'D'])
        return df

    def create_hierarchical_df(self, module):
        """
        Return a hierarchical DataFrame.
        """
        index_names = pd.MultiIndex.from_tuples([('Level1', 'Lev1', 'L1'),
                                                 ('Level2', 'Lev2', 'L2'),
                                                 ('Level3', 'Lev3', 'L3'),
                                                 ('Level4', 'Lev4', 'L4'),
                                                 ('Level5', 'Lev5', 'L5'),
                                                 ('Level6', 'Lev6', 'L6'),
                                                 ('Level7', 'Lev7', 'L7'),
                                                 ('Level8', 'Lev8', 'L8'),
                                                 ('Level9', 'Lev9', 'L9')],
                                                names=['Full', 'Partial', 'ID'])
        data = {'Store': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
                'Sales': [12, 44, 29, 35, 18, 27, 36, np.nan, 48],
                'Num': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        df = module.DataFrame(
            data, columns=['Store', 'Sales', 'Num'], index=index_names)
        return df

    def create_single_column_df(self, module, create_cache=False):
        """
        Return a single colunm DataFrame.
        """
        name = os.path.join(DATA_DIR, "single_column_df.csv")
        if self.use_csv and not create_cache:
            df = module.read_csv(name)
        else:
            data = np.arange(self.rows, 0, -1)
            data = data * 3
            df = module.DataFrame(data)
            if create_cache:
                df.to_csv(name)
        return df

    def create_single_row_df(self, module, create_cache=False):
        """
        Return a single row DataFrame.
        """
        name = os.path.join(DATA_DIR, "single_row_df.csv")
        if self.use_csv and not create_cache:
            df = module.read_csv(name)
        else:
            data = np.arange(self.rows, 0, -1)
            data = np.array([data * 3]).T
            df = module.DataFrame(data)
            if create_cache:
                df.to_csv(name)
        return df

    def create_df_gb_frame(self, module, create_cache=False):
        """
        Return a DataFrame for groupby comparison
        """
        # column 0 has all unique values, column 1 has num_gb_rows unique values
        name = os.path.join(DATA_DIR, "gb_frame.csv")
        if self.use_csv and not create_cache:
            df = module.read_csv(name)
        else:
            data = np.arange(self.rows * self.columns, 0, -1).reshape(self.rows, self.columns)
            data = data * 3
            num_gb_rows = 5
            for i in range(num_gb_rows):
                data[i::num_gb_rows, 1] = 1000 + i
            df = module.DataFrame(data)
            if create_cache:
                df.to_csv(name)
        return df

    def create_df_reindex(self, module):
        """
        Return a DataFrame for reindex purpose.
        """
        index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
        df = module.DataFrame({'http_status': [200, 200, 404, 404, 301],
                               'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
                              index=index)
        return df

    def create_df_setindex(self, module):
        """
        Return a DataFrame for setindex purpose.
        """
        df = module.DataFrame({'month': [1, 4, 7, 10],
                               'year': [2012, 2014, 2013, 2014],
                               'sale': [55, 40, 84, 31]})
        return df

    def create_df_duplicates(self, module):
        """
        Return a DataFrame that has duplicated rows.
        """
        df = module.DataFrame({'A': [2, 2, 2, 3, 3],
                               'B': [2, 2, 3, 2, 2],
                               'C': [2, 2, 1, 3, 3],
                               'D': [1, 1, 1, 3, 3]})
        return df

    def create_two_dfs(self, module):
        """
        Return two DataFrames.
        """
        df1 = module.DataFrame({'A': [1, 2, 3, 4],
                                'B': [5, 6, 7, 8],
                                'C': [9, 10, 11, 12],
                                'D': [13, 14, 15, 16]})
        df2 = module.DataFrame({'A': [1, 2, 3, 4],
                                'B': [5, 6, 7, 8],
                                'C': [9, 10, 11, 12],
                                'D': [13, 14, 15, 16]})
        return df1, df2

    def create_two_dfs_with_different_shape(self, module):
        """
        Return two DataFrames that have different shape.
        """
        df1 = module.DataFrame({'A': [1, 2, 3, 4],
                                'B': [5, 6, 7, 8],
                                'C': [9, 10, 11, 12]})
        df2 = module.DataFrame({'A': [1, 2, 3],
                                'B': [4, 5, 6],
                                'C': [7, 8, 9],
                                'D': [10, 11, 12]})
        return df1, df2

    def create_two_dfs_with_different_index(self, module):
        """
        Return two DataFrames that have different index.
        """
        df1 = module.DataFrame({'A': [1, 2, 3, 4],
                                'B': [5, 6, 7, 8],
                                'C': [9, 10, 11, 12],
                                'D': [13, 14, 15, 16]}, index=[0, 1, 2, 3])
        df2 = module.DataFrame({'E': [1, 2, 3, 4],
                                'F': [5, 6, 7, 8],
                                'G': [9, 10, 11, 12],
                                'H': [13, 14, 15, 16]}, index=[4, 5, 6, 7])
        return df1, df2

    def create_two_dfs_large(self, module):
        """
        Return two large size DataFrames.
        """
        np.random.seed(100)
        df1 = module.DataFrame(np.random.randint(1000, size=(1000, 1000)))
        np.random.seed(200)
        df2 = module.DataFrame(np.random.randint(1000, size=(1000, 1000)))
        return df1, df2

    def create_df_with_columns_and_index(self, module):
        """
        Return a DataFrame with columns and index.
        """
        data = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        index = ["index1", "index2"]
        columns = ["columns1", "columns2", "columns3", "columns4"]
        df = module.DataFrame(data, index, columns)
        return df

    def create_df_and_series(self, module):
        """
        Return a DataFrame and a Series.
        """
        df = module.DataFrame({'A': [1, 2, 3, 4],
                               'B': [5, 6, 7, 8]})
        series = module.Series([1, 2, 3, 4], name='A')
        return df, series

    def create_series_small(self, module):
        """
        Return a small size Series.
        """
        series = module.Series([1, 2, 3, 4], name='A')
        return series

    def create_ser_with_index(self, module):
        """
        Return a Series with index.
        """
        data = np.arange(5)
        ser = module.Series(data, name="value", index=list("abcde"))
        return ser

    def create_df_and_list(self, module):
        """
        Return a DataFrame and a list.
        """
        df = module.DataFrame({'A': [1, 2, 3, 4],
                               'B': [5, 6, 7, 8],
                               'C': [9, 10, 11, 12],
                               'D': [13, 14, 15, 16]})
        l = [1, 2, 3, 4]
        return df, l

    def create_df_and_scalar(self, module):
        """
        Return a DataFrame and a scalar.
        """
        df = module.DataFrame({'A': [1, 2, 3, 4],
                               'B': [5, 6, 7, 8],
                               'C': [9, 10, 11, 12],
                               'D': [13, 14, 15, 16]})
        scalar = 10
        return df, scalar

    def create_df_and_hierarchical_df(self, module):
        """
        Return a DataFrame and a hierarchical DataFrame.
        """
        df = module.DataFrame({'angles': [0, 3, 4],
                               'degrees': [360, 180, 360]},
                              index=['circle', 'triangle', 'rectangle'])
        df_multindex = module.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
                                         'degrees': [360, 180, 360, 360, 540, 720]},
                                        index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                               ['circle', 'triangle', 'rectangle',
                                                'square', 'pentagon', 'hexagon']])

        return df, df_multindex

    def create_series_range(self, module):
        """
        Return a Series that values from 0 to 999.
        """
        ser = module.Series(range(1000))
        return ser

    def create_hierarchical_series(self, module):
        """
        Return a hierarchical Series.
        """
        idx = pd.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'],
                                         ['fox', 'lion', 'snake', 'spider']],
                                        names=['blooded', 'animal'])
        ser = module.Series([4, 4, 0, 8], name="legs", index=idx)
        return ser

    def create_series_large(self, module):
        """
        Return a large Series with size 100000.
        """
        np.random.seed(100)
        ser = module.Series(np.random.randint(0, 100, size=100000))
        return ser

    def create_series_unique(self, module):
        """
        Return a Series only has one value <1>.
        """
        ser = module.Series([1])
        return ser

    def create_series_dup(self, module):
        """
        Return a Series with duplicated values in it.
        """
        ser = module.Series([3, 1, 2, 3, 4, np.nan])
        return ser

    def create_series_bool(self, module):
        """
        Return a Series that all values(1000 values) are boolean.
        """
        bools = [True, False, False, False, True, False, True, True]
        ser = module.Series(bools)
        return ser

    def create_series_nan(self, module):
        """
        Return a Series than all values are numpy.nan.
        """
        ser = module.Series([np.nan] * self.rows)
        return ser

    def create_series_zero(self, module):
        """
        Return a Series that all values are 0.
        """
        ser = module.Series([0] * self.rows)
        return ser

    def create_series_mixed_dtype(self, module):
        """
        Return a Series with mixed dtype.
        """
        ser = module.Series(["a", 1, 2, 3, np.nan])
        return ser

    def create_series_complex(self, module):
        """
        Return a Series with complex numbers
        """
        ser = module.Series([1 + 4j, 1 - 3j, 2, -4 - 4j, 2j, -3, -4])
        return ser

    def create_two_series(self, module):
        """
        Return two Series that have same index.
        """
        ser1 = module.Series([1, 1, 1, np.nan],
                             index=['a', 'b', 'c', 'd'])
        ser2 = module.Series([1, np.nan, 1, np.nan],
                             index=['a', 'b', 'c', 'd'])
        return ser1, ser2

    def create_two_series_with_different_index(self, module):
        """
        Return two Series that have different index.
        """
        ser1 = module.Series([1, 1, 1, np.nan],
                             index=['a', 'b', 'c', 'd'])
        ser2 = module.Series([1, np.nan, 1, np.nan],
                             index=['a', 'b', 'd', 'e'])
        return ser1, ser2

    def create_two_series_with_different_name(self, module):
        """
        Return two Series that have different name.
        """
        ser1 = module.Series([200, 250, 300, 350, 400],
                             index=['a', 'b', 'c', 'd', 'e'],
                             name="addtest")
        ser2 = pd.Series([100, 200, 300, 400, 500],
                         index=['a', 'b', 'c', 'd', 'e'],
                         name="test")
        return ser1, ser2

    def create_two_series_with_same_name(self, module):
        """
        Return two Series that have same name.
        """
        ser1 = module.Series([200, 250, 300, 350, 400],
                             index=['a', 'b', 'c', 'd', 'e'],
                             name="test")
        ser2 = pd.Series([100, 200, 300, 400, 500],
                         index=['a', 'b', 'c', 'd', 'e'],
                         name="test")
        return ser1, ser2

    def create_two_series_with_none_name(self, module):
        """
        Return two Series that one has name and one does not have name.
        """
        ser1 = module.Series([200, 250, 300, 350, 400],
                             index=['a', 'b', 'c', 'd', 'e'],
                             name="test")
        ser2 = pd.Series([100, 200, 300, 400, 500],
                         index=['a', 'b', 'c', 'd', 'e'])
        return ser1, ser2

    def create_series_and_hierarchical_series(self, module):
        """
        Return two Series that one of them is hierarchical Series.
        """
        ser1 = module.Series([1, 1, 2, 0],
                             index=['fox', 'lion', 'snake', 'spider'])
        idx = pd.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'],
                                         ['fox', 'lion', 'snake', 'spider']],
                                        names=['blooded', 'animal'])
        ser2 = module.Series([4, 4, 0, 8], name="legs", index=idx)
        return ser1, ser2

    def create_input_hierarchical_series_with_nan(self, module):
        """
        Return the Multi-index Series with nan value.
        Test <level>, <fill_value> purpose.
        """
        arrays = [['c1', 'c1', 'c2', 'c2', 'c3', 'c3'],
                  ['a', 'b', 'c', 'd', 'e']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["index1", "index2"])
        input_series = module.Series([1, np.nan, 1, np.nan, 1], index=index)
        return input_series

    def create_another_hierarchical_series_with_nan(self, input_series):
        """
        Return the same type Multi-index Series as <input_series> with nan value.
        Test <level>, <fill_value> purpose.
        """
        arrays = [['c1', 'c1', 'c2', 'c2', 'c3', 'c3'],
                  ['a', 'b', 'c', 'd', 'e']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["index1", "index2"])
        if isinstance(input_series, pd.Series):
            return pd.Series([0, 1, 2, np.nan, 1], index=index)
        return mpd.Series([0, 1, 2, np.nan, 1], index=index)

    def run_compare(self, function, create_fn=None, prep_fn=None, **kwargs):
        """
        The built-in function and performance compare functions.
        """
        if create_fn is None:
            create_fn = self.default_create_fn

        module = pd
        df = create_fn(module)
        if prep_fn is not None:
            df = prep_fn(df)
        df = function(df)

        module = mpd
        mdf = create_fn(module)
        if prep_fn is not None:
            mdf = prep_fn(mdf)
        mdf = function(mdf)

        if hasattr(mdf, "to_pandas"):
            mdf = mdf.to_pandas()

        if isinstance(df, pd.DataFrame):
            # TODO: fix dtype and name mismatch
            kwargs["check_dtype"] = False
            kwargs["check_names"] = False
            assert_frame_equal(df, mdf, **kwargs)
        elif isinstance(df, pd.Series):
            # TODO: fix dtype and name mismatch
            kwargs["check_dtype"] = False
            kwargs["check_names"] = False
            assert_series_equal(df, mdf, **kwargs)
        elif isinstance(df, pd.Index):
            assert_index_equal(df, mdf, **kwargs)
        elif isinstance(df, dict):
            assert_dict_equal(df, mdf, **kwargs)
        elif is_list_like(df):
            np.array_equal(df, mdf, **kwargs)
        else:
            # compare scalar
            assert isinstance(mdf, type(df))
            assert df == mdf

    def run_compare_error(self, err_function, err_type, create_fn):
        """
        General test function for test the correct error output
        """
        df = create_fn(pd)
        with pytest.raises(err_type):
            err_function(df)
        ms_df = create_fn(mpd)
        with pytest.raises(err_type):
            err_function(ms_df)

    def run_all_compare_time(self, function, create_fn=None, prep_fn=None):
        """
        The built-in function and performance compare functions with timing.
        """
        os.environ["MODIN_ENGINE"] = "ray"

        import modin.config
        import modin.pandas as mo_pd

        modin.config.BenchmarkMode.put(True)

        if create_fn is None:
            create_fn = self.default_create_fn

        modules = [(pd, "original_pandas"), (mpd, "mindpandas"), (mo_pd, "modin")]

        for module, name in modules:
            create_start = time.time()
            df = create_fn(module)
            if module is mpd and hasattr(df, "backend_frame"):
                df.backend_frame.flush()
            create_end = time.time()
            if prep_fn is not None:
                df = prep_fn(df)
            func_start = time.time()
            df = function(df)
            if module is mpd and hasattr(df, "backend_frame"):
                df.backend_frame.flush()
            func_end = time.time()
            print("{0}, {1}, create time:{2}, function time:{3}".format(function.__name__,
                                                                        name,
                                                                        create_end - create_start,
                                                                        func_end - func_start))

    def run_all_compare_time_groupby(self, function, create_fn=None, prep_fn=None):
        """
        The built in function and performance compare functions for method <groupby>.
        """
        if create_fn is None:
            create_fn = self.default_create_fn
        module = mpd
        # module_name = "mindspore.pandas"
        modules = [(pd, "original_pandas"), (mpd, "mindspore.pandas")]
        print("TESTING FN: ", function.__name__)
        print("create function is", create_fn.__name__)
        print("function_name, module, create_time, function_time, prep_time,total_time")
        have_orig_df = False
        for module, name in modules:
            t1 = time.time()
            df = create_fn(module)
            t2 = time.time()
            if prep_fn is not None:
                df = prep_fn(df)
            t3 = time.time()
            df = function(df)
            t4 = time.time()
            create_time = t2 - t1
            prep_time = t3 - t2
            function_time = t4 - t3
            total_time = t4 - t1
            print("##", function.__name__, ",", name, ",", create_time,
                  ",", function_time, ",", prep_time, ",", total_time)
            if not ('plot' in function.__name__ or 'hist' in function.__name__):
                if have_orig_df:
                    if function.__name__ == 'groupby_resample':
                        assert df_orig.equals(df)
                    elif isinstance(df_orig, pd.Series):
                        assert np.all(np.equal(df_orig.to_numpy(), df.to_numpy()))
                        # assert(df_orig.equals(df.backend_frame))
                    else:
                        assert df_orig.equals(df.to_pandas())
                else:
                    have_orig_df = True
                    df_orig = df

    def run_compare_multiple_df(self, function, create_fn=None, prep_fn=None, **kwargs):
        """
        The built-in function and performance compare functions for multiple DataFrame.
        """
        if create_fn is None:
            create_fn = self.default_create_fn

        module = pd
        df = create_fn(module)
        if prep_fn is not None:
            df = prep_fn(df)
        df_list = function(df)

        module = mpd
        mdf = create_fn(module)
        if prep_fn is not None:
            mdf = prep_fn(mdf)
        mdf_list = function(mdf)

        assert len(df_list) == len(mdf_list)

        for a, b in zip(df_list, mdf_list):
            if isinstance(a, pd.DataFrame):
                assert_frame_equal(a, b.to_pandas(), **kwargs)
            else:
                assert_series_equal(a, b.to_pandas(), **kwargs)

    def run_compare_error_special(self, err_function, err_type, create_fn, opt):
        """
        General test function for test the correct error output.
        """
        df = create_fn(pd)
        with pytest.raises(err_type):
            err_function(df, opt)
        ms_df = create_fn(mpd)
        with pytest.raises(err_type):
            err_function(ms_df, opt)


TESTUTIL = TestUtil()
