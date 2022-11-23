# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np

import pandas as pd
import mindpandas as mpd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_series_setitem():
    """
    Test Series setitem
    Description: tests Series.__setitem__
    Expectation: same output as pandas Series.__setitem__
    """

    def test_int_fn(ser):
        ser[1] = 5
        return ser

    def test_str_fn(ser):
        ser['a'] = 9
        return ser

    def test_array_fn(ser):
        ser[['c', 'd']] = [8, 9]
        return ser

    def test_array_unordered_fn(ser):
        ser[['d', 'c']] = [8, 9]
        return ser

    def test_slice_fn(ser):
        ser[1:3] = [8, 9]
        return ser

    TESTUTIL.compare(test_int_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_str_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_array_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_array_unordered_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_slice_fn, TESTUTIL.create_ser_with_index)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_setitem():
    """
    Test setitem
    Description: tests df.__setitem__
    Expectation: same output as pandas df.__setitem__
    """

    def test_setitem_column_wise_small(df):
        df['A'] = [0, 0, 0, 0, 0, 0]
        return df

    def test_setitem_column_wise_ndarray(df):
        df[0] = np.random.rand(100, 1)
        return df

    def test_setitem_column_wise_series(df):
        df[0] = pd.Series(np.random.rand(100))
        return df

    def test_setitem_column_wise_series_ms(df):
        df[0] = mpd.Series(np.random.rand(100))
        return df

    def test_setitem_column_wise_dataframe(df):
        df[0] = pd.DataFrame(np.random.rand(100, 1))
        return df

    def test_setitem_column_wise_dataframe_ms(df):
        df[0] = mpd.DataFrame(np.random.rand(100, 1))
        return df

    def test_setitem_new_column(df):
        df['new'] = np.random.rand(100, 1)
        return df

    def test_setitem_series(ser):
        ser[0] = 0
        return ser

    def test_setitem_element_wise(df):
        df[0][0] = 0
        return df

    def test_setitem_to_series(df):
        df["A"] = df["B"]
        return df

    def test_setitem_to_const(df):
        df["C"] = 1
        return df

    def test_setitem_to_const_new_column(df):
        df['new'] = 1
        return df

    def create_two_dfs(module):
        df = module.DataFrame()
        df2 = module.DataFrame(np.ones((2, 2)))
        return df, df2

    def test_setitem_empty_df_to_series_fn(dfs):
        df, df2 = dfs
        df['new'] = df2[0]
        return df

    def test_setitem_by_slice(df):
        df[2:4] = 0
        return df

    def hash_item(val, item_size=10000000, offset=0):
        if isinstance(val, str):
            return abs(int(hashlib.sha256(val.encode('utf-8')).hexdigest(), 16)) % item_size
        return abs(hash(val)) % item_size + offset

    def test_setitem_with_list_key_scalar_value(df):
        col_list = [str(c) for c in range(len(df.columns)//2)]
        df[col_list] = 1
        return df

    def test_setitem_with_list_key_array_value(df):
        col_list = [str(c) for c in range(len(df.columns)//2)]
        df[col_list] = df[col_list].applymap(hash_item)
        return df


    TESTUTIL.compare(test_setitem_column_wise_small, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_setitem_column_wise_ndarray, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_setitem_column_wise_series, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_setitem_column_wise_series_ms, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_setitem_column_wise_dataframe, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_setitem_column_wise_dataframe_ms, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_setitem_new_column, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_setitem_series, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_setitem_series, create_fn=TESTUTIL.create_series_unique)
    TESTUTIL.compare(test_setitem_element_wise, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_setitem_to_series)
    TESTUTIL.compare(test_setitem_to_const)
    TESTUTIL.compare(test_setitem_to_const_new_column)
    TESTUTIL.compare(test_setitem_empty_df_to_series_fn, create_fn=create_two_dfs)
    TESTUTIL.compare(test_setitem_by_slice, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_setitem_with_list_key_scalar_value, create_fn=TESTUTIL.create_df_array_with_nan)
    TESTUTIL.compare(test_setitem_with_list_key_array_value, create_fn=TESTUTIL.create_df_array_with_nan)
