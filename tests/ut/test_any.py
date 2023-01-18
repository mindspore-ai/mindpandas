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
def test_any():
    """
    Test any
    Description: tests df.any
    Expectation: same output as pandas df.any
    """

    def test_any_fn(df):
        df = df.any()
        return df

    def test_any_axis_none(df):
        df = df.any(axis=None)
        return df

    def test_any_bool_only(df):
        df = df.any(bool_only=True)
        return df

    def test_any_level0(df):
        df = df.any(level=0)
        return df

    def test_any_level1(df):
        df = df.any(level=1)
        return df

    TESTUTIL.compare(test_any_fn, TESTUTIL.create_df_bool)
    TESTUTIL.compare(test_any_axis_none, TESTUTIL.create_df_bool)
    TESTUTIL.compare(test_any_bool_only, TESTUTIL.create_df_bool_and_str)

    TESTUTIL.compare(test_any_fn, TESTUTIL.create_series_bool)
    TESTUTIL.compare(test_any_fn, TESTUTIL.create_series_nan)
    TESTUTIL.compare(test_any_fn, TESTUTIL.create_series_zero)
    TESTUTIL.compare(test_any_level0, TESTUTIL.create_hierarchical_series)
    TESTUTIL.compare(test_any_level1, TESTUTIL.create_hierarchical_series)
