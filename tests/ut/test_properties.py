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
def test_properties():
    """
    Test DataFrame and Series' properties
    Description: tests df and series properties
    Expectation: same output as pandas
    """

    def test_get_index(df):
        return df.index

    def test_index_setter(df):
        df.index = [_ for _ in range(5, 10)]
        return df

    def test_get_columns(df):
        return df.columns

    def test_columns_setter(df):
        df.columns = ['D', 'C', 'B', 'A']
        return df

    def test_empty(df):
        return df.empty

    def test_shape(df):
        return df.shape

    def test_size(df):
        """
        Test size
        """
        return df.size

    def test_values(df):
        return df.values

    def create_df_0_index(module):
        df = module.DataFrame({'id': []})
        return df

    def create_df_single_value(module):
        df = module.DataFrame({0: [1]}, index=[0])
        return df

    def create_df_empty(module):
        df = module.DataFrame([])
        return df

    def create_df_empty2(module):
        df = module.DataFrame()
        return df

    def create_series_empty(module):
        ser = module.Series([])
        return ser

    TESTUTIL.compare(test_get_index, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_index_setter, TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_get_columns, TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_columns_setter, TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_empty, create_df_0_index)
    TESTUTIL.compare(test_empty, create_df_empty)
    TESTUTIL.compare(test_empty, create_df_empty2)
    TESTUTIL.compare(test_empty, create_df_single_value)
    TESTUTIL.compare(test_shape, TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_empty, create_series_empty)
    TESTUTIL.compare(test_empty, TESTUTIL.create_series_unique)
    TESTUTIL.compare(test_shape, TESTUTIL.create_series_range)
    TESTUTIL.compare(test_size, TESTUTIL.create_series_range)
    TESTUTIL.compare(test_values, TESTUTIL.create_series_unique)
