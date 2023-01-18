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

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_explode():
    """
    Description: tests df.explode
    Expectation: same output as pandas df.explode
    """

    def create_one_col_empty(module):
        df = module.DataFrame({'A': []})
        return df

    def create_one_col_single_element(module):
        df = module.DataFrame({'A': [9]})
        return df

    def create_multiple_lists_df(module):
        np.random.seed(100)
        y = np.random.randint(0, 10, (2000))
        df = module.DataFrame({'A': [2, 4, 6, [8, 10]],
                               'B': ['qwerty', y[:1500], [], y[4:892]],
                               'C': ['z', y[500:], [], y[1112:]],
                               'E': 9,
                               'D': [np.nan, y[200:1700], [], y[:888]]})
        return df

    def test_explode_single_column(df):
        df = df.explode(['A'])
        return df

    def test_explode_single_column_ignore_index(df):
        df = df.explode(['A'], ignore_index=True)
        return df

    def test_explode_multiple_column(df):
        df = df.explode(list('BCD'))
        return df

    def test_explode_multiple_column_ignore_index(df):
        df = df.explode(list('BCD'), ignore_index=True)
        return df

    TESTUTIL.compare(test_explode_single_column, create_one_col_empty)
    TESTUTIL.compare(test_explode_single_column, create_one_col_single_element)
    TESTUTIL.compare(test_explode_single_column, create_multiple_lists_df)
    TESTUTIL.compare(test_explode_single_column_ignore_index, create_multiple_lists_df)
    TESTUTIL.compare(test_explode_multiple_column, create_multiple_lists_df)
    TESTUTIL.compare(test_explode_multiple_column_ignore_index, create_multiple_lists_df)
    TESTUTIL.compare(test_explode_single_column, TESTUTIL.create_df_range)
    TESTUTIL.compare(test_explode_single_column_ignore_index, TESTUTIL.create_df_range)
    TESTUTIL.compare(test_explode_multiple_column, TESTUTIL.create_df_range)
    TESTUTIL.compare(test_explode_multiple_column_ignore_index, TESTUTIL.create_df_range)
