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
def test_sort_values():
    """
    Test sort_values
    Description: tests df.sort_values
    Expectation: same output as pandas df.sort_values
    """

    def test_sort_values_default(df):
        df = df.sort_values(by=['A', 'B', 'C'])
        return df

    def test_sort_values_axis(df):
        df = df.sort_values(by=['A', 'B', 'C'], axis=0)
        return df

    def test_sort_values_descending(df):
        df = df.sort_values(by=['A', 'B', 'C'], ascending=False)
        return df

    def test_sort_values_inplace(df):
        df = df.sort_values(by=['A', 'B', 'C'], inplace=True)
        return df

    def test_sort_values_mergesort(df):
        df = df.sort_values(by=['A', 'B', 'C'], kind='mergesort')
        return df

    def test_sort_values_heapsort(df):
        df = df.sort_values(by=['A', 'B', 'C'], kind='heapsort')
        return df

    def test_sort_values_stable(df):
        df = df.sort_values(by=['A', 'B', 'C'], kind='stable')
        return df

    def test_sort_values_na_position(df):
        df = df.sort_values(by=['A', 'B', 'C'], na_position='first')
        return df

    def test_sort_values_na_ignore_index(df):
        df = df.sort_values(by=['A', 'B', 'C'], ignore_index=True)
        return df

    def test_sort_values_multiindex(df):
        df = df.sort_values(by="Sales")
        return df

    def test_sort_values_non_numeric(df):
        df = df.sort_values(by="City")
        return df

    TESTUTIL.compare(test_sort_values_default, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_axis, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_descending, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_inplace, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_mergesort, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_heapsort, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_stable, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_na_position, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_na_ignore_index, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sort_values_default, create_fn=TESTUTIL.create_df_has_duplicate_index)
    TESTUTIL.compare(test_sort_values_inplace, create_fn=TESTUTIL.create_df_has_duplicate_index)
    TESTUTIL.compare(test_sort_values_na_position, create_fn=TESTUTIL.create_df_has_duplicate_index)
    TESTUTIL.compare(test_sort_values_na_ignore_index,
                     create_fn=TESTUTIL.create_df_has_duplicate_index)
    TESTUTIL.compare(test_sort_values_multiindex, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_sort_values_non_numeric, create_fn=TESTUTIL.create_df_mixed_dtypes)
