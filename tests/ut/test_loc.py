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
def test_loc_getitem():
    """
    Test loc getitem
    Description: tests df.loc with getitem
    Expectation: same output as pandas df.loc with getitem
    """

    def test_loc_getitem_fn(df):
        df = df.loc[3:50, 'A':'D']
        return df

    TESTUTIL.compare(test_loc_getitem_fn, create_fn=TESTUTIL.create_df_range)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_loc_setitem():
    """
    Test loc setitem
    Description: tests df.loc with setitem
    Expectation: same output as pandas df.loc with setitem
    """

    def test_loc_setitem_to_integer_fn(df):
        df.loc[[2, 99], ['A', 'D']] = 100
        return df

    def test_loc_setitem_to_list_fn(df):
        df.loc[[2, 99], ['A', 'D']] = [100, 300]
        return df

    def test_loc_setitem_row_fn(df):
        df.loc[2] = 100
        return df

    def test_loc_setitem_column_fn(df):
        df.loc[:, 'A'] = 100
        return df

    def test_loc_setitem_condition_fn(df):
        df.loc[df['A'] > 3] = 100
        return df

    def test_loc_setitem_callable_fn(df):
        df.loc[lambda df: df['A'] < 1000, 'B'] = 100000000
        return df

    def test_loc_setitem_slice_fn(df):
        df.loc[0: 3, 'A': 'D'] = [100, 100, 100, 300]
        return df

    def test_loc_setitem_not_existing_column_fn(df):
        df.loc[:, 'F'] = 100
        return df

    def test_loc_setitem_to_series_fn(df):
        df.loc[:, "F"] = df['A']
        return df

    def test_loc_setitem_to_dataframe_fn(df):
        df.loc[:, "F"] = mpd.DataFrame([0] * 100)
        return df

    def test_loc_setitem_subset_dataframe_fn(df):
        df.loc[0: 3, 'A': 'D'] = pd.DataFrame(data=[0] * 4, columns=['A'])
        return df

    TESTUTIL.compare(test_loc_setitem_to_integer_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_to_list_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_row_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_column_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_condition_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_callable_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_slice_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_not_existing_column_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_to_series_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_to_dataframe_fn, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_loc_setitem_subset_dataframe_fn, create_fn=TESTUTIL.create_df_range)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_loc():
    """
    Test loc
    Description: tests df.loc
    Expectation: same output as pandas df.loc
    """

    def create_loc_default_to_pandas_test_frame(module):
        pandas_df = pd.DataFrame([[1, 2], [3, 4], [5, 6]],
                                 index=['row_a', 'row_b', 'row_c'],
                                 columns=['col_a', 'col_b'])
        if module == mpd:
            return mpd.DataFrame(data=pandas_df)
        return pandas_df

    def test_loc_with_list_of_labels(df):
        print(df.loc[['row_b', 'row_c']])
        return df.loc[['row_b', 'row_c']]

    def test_loc_with_row_slice(df):
        print(df.loc['row_a': 'row_b'])
        return df.loc['row_a': 'row_b']

    def test_loc_with_boolean_array(df):
        print(df.loc[[False, False, True]])
        return df.loc[[False, False, True]]

    TESTUTIL.compare(test_loc_with_list_of_labels,
                     create_fn=create_loc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_loc_with_row_slice, create_fn=create_loc_default_to_pandas_test_frame)
    TESTUTIL.compare(test_loc_with_boolean_array, create_fn=create_loc_default_to_pandas_test_frame)
