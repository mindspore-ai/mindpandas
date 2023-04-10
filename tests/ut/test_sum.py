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
def test_sum():
    """
    Test sum
    Description: tests df.sum
    Expectation: same output as pandas df.sum
    """

    def create_df_multiindex(module):
        idx = pd.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'],
                                         ['dog', 'cat', 'fish', 'spider']],
                                        names=['blooded', 'animal'])
        data = np.array([[None, None, 3, 4], [5, None, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        return module.DataFrame(data, index=idx)

    def create_series_multiindex(module):
        idx = pd.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold']], names=['blooded'])
        data = np.array([13, 14, 15, 16])
        return module.Series(data, index=idx)

    def create_df_non_numeric(module):
        return module.DataFrame({"C1": ["a", 10], "C2": [3, 4]})

    def create_df_non_numeric1(module):
        return module.DataFrame({"C1": ["a", 10], "C2": [3, 4], "C3": pd.Series([3, 4], dtype='float32')})

    def create_df_non_numeric2(module):
        return module.DataFrame({"C1": ["a", "b"], "C2": pd.Series([3, 4], dtype='float128'),
                                 "C3": pd.Series([3, 4], dtype='int64'), "C4": [3, 4.0]})

    def creat_issue_df(module):
        return module.DataFrame([["a", 1]])

    def test_sum_default(df):
        df = df.sum()
        return df

    def test_sum_axis_1(df):
        df = df.sum(axis=1)
        return df

    def test_sum_level_full(df):
        df = df.sum(axis=0, level='Full')
        return df

    def test_sum_level_partial(df):
        df = df.sum(axis=0, level='Partial')
        return df

    def test_sum_level_id(df):
        df = df.sum(axis=0, level='ID')
        return df

    def test_sum_numeric_only_is_true(df):
        return df.sum(numeric_only=True)

    def test_sum_numeric_only_is_false(df):
        return df.sum(numeric_only=False)

    def test_sum_numeric_only_is_string(df):
        return df.sum(numeric_only="Test")

    def test_sum_min_count(df):
        return df.sum(axis=0, level="animal", skipna=False, min_count=3)

    def test_sum_min_count_only(df):
        return df.sum(level="blooded", min_count=4)

    def test_sum_numeric_only_is_true_axis1(df):
        return df.sum(axis=1, skipna=False, numeric_only=True)

    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_df_empty)
    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_df_empty_with_columns)
    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_df_mixed_dtypes_2)

    TESTUTIL.compare(test_sum_axis_1, create_fn=TESTUTIL.create_df_empty)
    TESTUTIL.compare(test_sum_axis_1, create_fn=TESTUTIL.create_df_empty_with_columns)
    TESTUTIL.compare(test_sum_axis_1, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_sum_axis_1, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_sum_axis_1, create_fn=TESTUTIL.create_df_mixed_dtypes_2)

    TESTUTIL.compare(test_sum_level_full, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_sum_level_partial, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_sum_level_id, create_fn=TESTUTIL.create_hierarchical_df)

    TESTUTIL.compare(test_sum_numeric_only_is_true, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_sum_numeric_only_is_false, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_sum_numeric_only_is_string, create_fn=TESTUTIL.create_df_mixed_dtypes, skip=True)

    TESTUTIL.compare(test_sum_min_count, create_fn=create_df_multiindex)
    TESTUTIL.compare(test_sum_min_count_only, create_fn=create_series_multiindex)
    TESTUTIL.compare(test_sum_numeric_only_is_true_axis1, create_fn=create_df_non_numeric)
    TESTUTIL.compare(test_sum_numeric_only_is_true_axis1, create_fn=create_df_non_numeric1)
    TESTUTIL.compare(test_sum_numeric_only_is_true_axis1, create_fn=create_df_non_numeric2, skip=True)
    TESTUTIL.compare(test_sum_numeric_only_is_true_axis1, create_fn=creat_issue_df)

    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_series_bool)
    TESTUTIL.compare(test_sum_default, create_fn=TESTUTIL.create_series_nan)
