# Copyright 2023 Huawei Technologies Co., Ltd
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
def test_prod():
    """
    Test prod
    Description: tests df.prod
    Expectation: same output as pandas.DataFrame.prod
    """

    def test_prod_default(df):
        df = df.prod()
        return df

    def test_prod_axis_1(df):
        df = df.prod(axis=1)
        return df

    def test_prod_level_full(df):
        df = df.prod(axis=0, level='Full')
        return df

    def test_prod_level_partial(df):
        df = df.prod(axis=0, level='Partial')
        return df

    def test_prod_level_id(df):
        df = df.prod(axis=0, level='ID')
        return df

    def test_prod_level_blooded(df):
        df = df.prod(level='blooded')
        return df

    def test_prod_numeric_only_is_true(df):
        df = df.prod(numeric_only=True)
        return df

    def test_prod_skipna_is_false(df):
        df = df.prod(skipna=False)
        return df

    def test_prod_args_combination_1(df):
        df = df.prod(axis=1, numeric_only=True)
        return df

    def test_prod_args_combination_2(df):
        df = df.prod(axis=1, skipna=False, numeric_only=True)
        return df

    # DataFrame.prod
    TESTUTIL.compare(test_prod_default, create_fn=TESTUTIL.create_df_empty)
    TESTUTIL.compare(test_prod_default, create_fn=TESTUTIL.create_df_empty_with_columns)
    TESTUTIL.compare(test_prod_default, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_prod_level_full, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_prod_level_partial, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_prod_level_id, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_prod_default, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_prod_axis_1, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_prod_numeric_only_is_true, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_prod_skipna_is_false, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_prod_args_combination_1, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_prod_args_combination_2, create_fn=TESTUTIL.create_df_mixed_dtypes)

    # Series.prod
    TESTUTIL.compare(test_prod_default, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_prod_default, create_fn=TESTUTIL.create_series_bool)
    TESTUTIL.compare(test_prod_default, create_fn=TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_prod_level_blooded, create_fn=TESTUTIL.create_hierarchical_series)
