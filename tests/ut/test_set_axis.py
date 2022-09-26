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

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_set_axis():
    """
    Test set_axis
    Description: tests df.set_axis and series.set_axis
    Expectation: same output as pandas
    """

    def create_df(module):
        return module.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    def test_set_index_default(df):
        new_label = [_ for _ in range(5, 10)]
        df.set_axis(axis=0, labels=new_label)
        return df

    def test_set_index_inplace(df):
        new_label = [_ for _ in range(5, 10)]
        df.set_axis(axis=0, labels=new_label, inplace=True)
        return df

    def test_set_columns_default(df):
        new_label = ['D', 'C', 'B', 'A']
        df.set_axis(axis=1, labels=new_label)
        return df

    def test_set_columns_inplace(df):
        new_label = ['D', 'C', 'B', 'A']
        df.set_axis(axis=1, labels=new_label, inplace=True)
        return df

    def test_set_axis_series(series):
        new_index = [7, 8, 9, 10]
        return series.set_axis(new_index)

    def test_set_axis_inplace_series(series):
        new_index = [7, 8, 9, 10]
        series.set_axis(new_index, inplace=True)
        return series

    def test_df_axis_row(df):
        return df.set_axis(['a', 'b', 'c'], axis='index')

    def test_df_axis_col(df):
        return df.set_axis(['I', 'II'], axis='columns')

    def test_df_axis_inplace(df):
        new_df = df.set_axis(['i', 'ii'], axis='columns', inplace=True)
        return new_df

    TESTUTIL.compare(test_set_index_default, TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_set_index_inplace, TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_set_columns_default, TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_set_columns_inplace, TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_set_axis_series, create_fn=TESTUTIL.create_series_small)
    TESTUTIL.compare(test_set_axis_inplace_series, create_fn=TESTUTIL.create_series_small)
    TESTUTIL.compare(test_df_axis_row, create_fn=create_df)
    TESTUTIL.compare(test_df_axis_col, create_fn=create_df)
    TESTUTIL.compare(test_df_axis_inplace, create_fn=create_df)
