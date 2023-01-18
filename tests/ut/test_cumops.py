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
def test_cumsum():
    """
    Test cumsum
    Description: tests df.cumsum
    Expectation: same output as pandas df.cumsum
    """

    def test_cumsum_fn(df):
        df = df.cumsum()
        return df

    TESTUTIL.compare(test_cumsum_fn)
    TESTUTIL.compare(test_cumsum_fn, TESTUTIL.create_series_bool)
    TESTUTIL.compare(test_cumsum_fn, TESTUTIL.create_series_nan)
    TESTUTIL.compare(test_cumsum_fn, TESTUTIL.create_series_zero)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_cummin():
    """
    Test cummin
    Description: tests df.cummin
    Expectation: same output as pandas df.cummin
    """

    def test_cummin_fn(df):
        df = df.cummin(axis=0)
        return df

    TESTUTIL.compare(test_cummin_fn)
    TESTUTIL.compare(test_cummin_fn, TESTUTIL.create_series_bool)
    TESTUTIL.compare(test_cummin_fn, TESTUTIL.create_series_nan)
    TESTUTIL.compare(test_cummin_fn, TESTUTIL.create_series_zero)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_cummax():
    """
    Test cummax
    Description: tests df.cummax
    Expectation: same output as pandas df.cummax
    """

    def test_cummax_fn(df):
        df = df.cummax(axis=0)
        return df

    TESTUTIL.compare(test_cummax_fn)
    TESTUTIL.compare(test_cummax_fn, TESTUTIL.create_series_bool)
    TESTUTIL.compare(test_cummax_fn, TESTUTIL.create_series_nan)
    TESTUTIL.compare(test_cummax_fn, TESTUTIL.create_series_zero)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_cumprod():
    """
    Test cumprod
    Description: tests df.cumprod
    Expectation: same output as pandas df.cumprod
    """

    def test_cumprod_fn(df):
        df = df.cumprod(axis=0)
        return df

    TESTUTIL.compare(test_cumprod_fn)
    TESTUTIL.compare(test_cumprod_fn, TESTUTIL.create_series_bool)
    TESTUTIL.compare(test_cumprod_fn, TESTUTIL.create_series_nan)
    TESTUTIL.compare(test_cumprod_fn, TESTUTIL.create_series_zero)
