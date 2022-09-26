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
def test_max():
    """
    Test max
    Description: tests df.max
    Expectation: same output as pandas df.max
    """

    def test_max_fn(df):
        df = df.max()
        return df

    def test_max_numeric_only(df):
        df = df.max(numeric_only=True)
        return df

    TESTUTIL.compare(test_max_fn, TESTUTIL.create_df_gaussian)
    TESTUTIL.compare(test_max_numeric_only, TESTUTIL.create_df_int_and_str)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_series_max():
    """
    Test series max
    Description: tests series.max
    Expectation: same output as pandas series.max
    """

    def test_series_max_fn(ser):
        ser = ser.max()
        return ser

    def test_hierarchical_series_max_fn(ser):
        ser = ser.max(level='blooded')
        return ser

    TESTUTIL.compare(test_series_max_fn, TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_series_max_fn, TESTUTIL.create_series_range)
    TESTUTIL.compare(test_series_max_fn, TESTUTIL.create_series_unique)
    TESTUTIL.compare(test_series_max_fn, TESTUTIL.create_series_large)
    TESTUTIL.compare(test_hierarchical_series_max_fn, TESTUTIL.create_hierarchical_series)
