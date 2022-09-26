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

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_getitem():
    """
    Test DataFrame getitem
    Description: tests df.__getitem__
    Expectation: same output as pandas df.__getitem__
    """

    def test_getitem_by_name(df):
        output = df['A']
        return output

    def test_getitem_by_list(df):
        output = df[['A', 'B']]
        return output

    def test_getitem_by_list_unordered(df):
        output = df[['B', 'A']]
        return output

    def test_getitem_by_array(df):
        output = df[np.array(['A', 'B'])]
        return output

    def test_getitem_by_index(df):
        output = df[pd.Index(['A', 'B'])]
        return output

    def test_getitem_by_series(df):
        output = df[pd.Series(['A', 'B'])]
        return output

    def test_getitem_by_dataframe(df):
        output = df[pd.DataFrame(['A', 'B'])]
        return output

    def test_getitem_by_list_boolean(df):
        output = df[[True, False, True, False, True, False]]
        return output

    def test_getitem_by_np_boolean(df):
        output = df[np.array([True, False, True, False, True, False])]
        return output

    def test_getitem_by_series_boolean(df):
        output = df[pd.Series([True, False, True, False, True, False])]
        return output

    def test_getitem_by_series_boolean_all_false(df):
        output = df[pd.Series([True, False, True, False, True, False])]
        return output

    def test_getitem_by_slice(df):
        output = df[2:]
        return output

    TESTUTIL.compare(test_getitem_by_name, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_list, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_list_unordered, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_array, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_index, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_series, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_dataframe, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_list_boolean, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_np_boolean, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_series_boolean, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_series_boolean_all_false, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_getitem_by_slice, create_fn=TESTUTIL.create_df_small)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_series_getitem():
    """
    Test Series getitem
    Description: tests Series.__getitem__
    Expectation: same output as pandas Series.__getitem__
    """

    def test_str_fn(ser):
        ser = ser['c']
        return ser

    def test_int_fn(ser):
        ser = ser[2]
        return ser

    def test_slice_fn(ser):
        ser = ser[2:4]
        return ser

    def test_array_fn(ser):
        ser = ser[['c', 'd']]
        return ser

    def test_array_unordered_fn(ser):
        ser = ser[['d', 'c']]
        return ser

    def test_bool_indexer(ser):
        ser = ser[[True, False, True, False, True]]
        return ser

    TESTUTIL.compare(test_str_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_int_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_slice_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_array_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_array_unordered_fn, TESTUTIL.create_ser_with_index)
    TESTUTIL.compare(test_bool_indexer, TESTUTIL.create_ser_with_index)
