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
def test_math_operations():
    """
    Test math operation (eg. add, sub, mul, div)
    Description: tests df with math ops
    Expectation: same output as pandas for those ops
    """

    def test_math_op_default(dfs):
        df1, df2 = dfs
        return getattr(df1, operation)(df2)

    def test_math_op_axis0(dfs):
        df1, df2 = dfs
        return getattr(df1, operation)(df2, axis=0)

    def test_math_op_level0(dfs):
        df1, df2 = dfs
        return getattr(df1, operation)(df2, level=0)

    def test_math_op_level1(dfs):
        df1, df2 = dfs
        return getattr(df1, operation)(df2, level=1)

    def test_math_op_fill_value1(dfs):
        df1, df2 = dfs
        return getattr(df1, operation)(df2, fill_value=1)

    def test_math_op_series_scalar(ser):
        return getattr(ser, operation)(1)

    def test_math_op_ser_from_df(df_scalar):
        df, _ = df_scalar
        return getattr(df["A"], operation)(df["B"])

    for operation in ["add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow"]:
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_dfs)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_dfs_large)

        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_dfs_with_different_shape)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_dfs_with_different_index)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_df_and_series)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_df_and_list)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_df_and_scalar)

        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_series_with_different_name)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_series_with_same_name)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_series_with_none_name)

        TESTUTIL.compare(test_math_op_fill_value1, TESTUTIL.create_two_dfs_with_different_shape)
        TESTUTIL.compare(test_math_op_fill_value1, TESTUTIL.create_two_dfs_with_different_index)

        TESTUTIL.compare(test_math_op_axis0, TESTUTIL.create_two_dfs_with_different_shape)
        TESTUTIL.compare(test_math_op_axis0, TESTUTIL.create_two_dfs_with_different_index)
        TESTUTIL.compare(test_math_op_axis0, TESTUTIL.create_df_and_series)
        TESTUTIL.compare(test_math_op_axis0, TESTUTIL.create_df_and_list)

        TESTUTIL.compare(test_math_op_level0, TESTUTIL.create_df_and_hierarchical_df)
        TESTUTIL.compare(test_math_op_level1, TESTUTIL.create_df_and_hierarchical_df)

        TESTUTIL.compare(test_math_op_series_scalar, TESTUTIL.create_series_range)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_series)
        TESTUTIL.compare(test_math_op_default, TESTUTIL.create_two_series_with_different_index)
        TESTUTIL.compare(test_math_op_fill_value1, TESTUTIL.create_two_series)
        TESTUTIL.compare(test_math_op_level0, TESTUTIL.create_series_and_hierarchical_series)
        TESTUTIL.compare(test_math_op_level1, TESTUTIL.create_series_and_hierarchical_series)
        TESTUTIL.compare(test_math_op_ser_from_df, TESTUTIL.create_df_and_scalar)
