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
def test_duplicated():
    """
    Test duplicated
    Description: tests df.duplicated
    Expectation: same output as pandas df.duplicated
    """

    def test_duplicated_all(df):
        df = df.duplicated()
        return df

    def test_duplicated_one_column_a(df):
        df = df.duplicated(subset=['A'])
        return df

    def test_duplicated_two_column_bc(df):
        df = df.duplicated(subset=['B', 'C'])
        return df

    def create_input_dataframe(module):
        df = module.DataFrame({
            'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
            'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
            'rating': [4, 4, 3.5, 15, 5]})
        return df

    def test_duplicated_fn(df):
        df = df.duplicated()
        return df

    def test_duplicated_keep_last(df):
        df = df.duplicated(keep='last')
        return df

    def test_duplicated_keep_false(df):
        df = df.duplicated(keep=False)
        return df

    def test_duplicated_subset_string(df):
        df = df.duplicated(subset='brand')
        return df

    def test_duplicated_subset_list(df):
        df = df.duplicated(subset=['brand'])
        return df

    def test_duplicated_subset_tuple(df):
        df = df.duplicated(subset=('brand', 'rating'))
        return df

    def test_err_duplicated_subset(df):
        df.duplicated(subset=1)

    TESTUTIL.compare(test_duplicated_all, create_fn=TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_duplicated_one_column_a, create_fn=TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_duplicated_two_column_bc, create_fn=TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_duplicated_fn, create_fn=create_input_dataframe)
    TESTUTIL.compare(test_duplicated_keep_last, create_fn=create_input_dataframe)
    TESTUTIL.compare(test_duplicated_keep_false, create_fn=create_input_dataframe)
    TESTUTIL.compare(test_duplicated_subset_string, create_fn=create_input_dataframe)
    TESTUTIL.compare(test_duplicated_subset_list, create_fn=create_input_dataframe)
    TESTUTIL.compare(test_duplicated_subset_tuple, create_fn=create_input_dataframe)
    TESTUTIL.run_compare_error(test_err_duplicated_subset, KeyError, create_input_dataframe)
