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

import pandas as pd
import mindpandas as mpd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_iloc_getitem():
    """
    Test iloc getitem
    Description: tests df.iloc with getitem
    Expectation: same output as pandas df.iloc with getitem
    """

    def test_iloc_getitem_fn(df):
        df = df.iloc[3:50, 0:3]
        return df

    TESTUTIL.compare(test_iloc_getitem_fn, create_fn=TESTUTIL.create_df_range)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_iloc_setitem():
    """
    Test iloc setitem
    Description: tests df.iloc with setitem
    Expectation: same output as pandas df.iloc with setitem
    """

    def test_loc_setitem_scalar_fn(df):
        df.iloc[0] = 100
        return df

    def test_iloc_setitem_list_fn(df):
        df.iloc[[0, 2]] = 100
        return df

    def test_iloc_setitem_slice_fn(df):
        df.iloc[:3] = 100
        return df

    TESTUTIL.compare(test_loc_setitem_scalar_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_iloc_setitem_list_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_iloc_setitem_slice_fn, create_fn=TESTUTIL.create_df_range)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_iloc():
    """
    Test iloc
    Description: tests df.iloc
    Expectation: same output as pandas df.iloc
    """

    def create_iloc_default_to_pandas_test_frame(module):
        pandas_df = pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4},
                                  {'a': 5, 'b': 6, 'c': 7, 'd': 8},
                                  {'a': 9, 'b': 10, 'c': 11, 'd': 12}])
        if module == mpd:
            return mpd.DataFrame(data=pandas_df)
        return pandas_df

    def test_iloc_with_list_of_integers(df):
        return df.iloc[[0, 1]]

    def test_iloc_with_slice(df):
        return df.iloc[0: 2]

    def test_iloc_with_boolean_array(df):
        return df.iloc[[True, False, True]]

    def test_iloc_for_row_and_column_1(df):
        return df.iloc[[1, 2], [1, 3]]

    def test_iloc_for_row_and_column_2(df):
        return df.iloc[1:3, 0:2]

    def test_iloc_default1(df):
        return df.iloc[:2]

    def test_iloc_default2(df):
        return df.iloc[-2:]

    def test_iloc_default3(df):
        return df.iloc[:]

    def test_iloc_default4(df):
        return df.iloc[::-1]

    def test_iloc_default5(df):
        return df.iloc[5:8]

    def test_iloc_default6(df):
        return df.iloc[1:8]

    def test_iloc_default7(df):
        return df.iloc[-5:]

    def test_head(df):
        return df.head()

    def test_tail(df):
        return df.tail()

    # ====================DataFrame Testcases====================
    TESTUTIL.compare(test_iloc_with_list_of_integers,
                     create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_with_slice, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_with_boolean_array,
                     create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_for_row_and_column_1,
                     create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_for_row_and_column_2,
                     create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_default1, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_default2, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_default3, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_default4, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_default5, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_default6, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_iloc_default7, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_head, create_fn=create_iloc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_tail, create_fn=create_iloc_default_to_pandas_test_frame)
    # ====================Series Testcases====================
    TESTUTIL.compare(test_iloc_with_list_of_integers, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_iloc_with_slice, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_iloc_default1, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_iloc_default2, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_iloc_default3, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_iloc_default4, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_iloc_default5, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_iloc_default6, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_iloc_default7, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_head, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_tail, create_fn=TESTUTIL.create_series_range)
