# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pandas as pd
import pytest

import mindpandas as mpd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_df_with_df():
    """
    Test DataFrame.fillna() with a DataFrame as an input value
    Description: tests df.fillna(df)
    Expectation: same output as pandas df.fillna(df)
    """
    df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                       [3, 4, np.nan, 1],
                       [np.nan, np.nan, np.nan, np.nan],
                       [np.nan, 3, np.nan, 4]],
                      columns=['a', 'b', 'c', 'd'])
    ms_df = mpd.DataFrame([[np.nan, 2, np.nan, 0],
                           [3, 4, np.nan, 1],
                           [np.nan, np.nan, np.nan, np.nan],
                           [np.nan, 3, np.nan, 4]],
                          columns=['a', 'b', 'c', 'd'])
    values_df_zero = pd.DataFrame(np.zeros((4, 4)), columns=['a', 'b', 'c', 'd'])
    ms_values_df_zero = mpd.DataFrame(np.zeros((4, 4)), columns=['a', 'b', 'c', 'd'])
    assert df.fillna(values_df_zero).equals(ms_df.fillna(ms_values_df_zero).to_pandas())


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_ser_with_ser():
    """
    Test Series.fillna() with a Series as an input value
    Description: tests Series.fillna(ser)
    Expectation: same output as pandas Series.fillna(ser)
    """
    ser = pd.Series([np.nan, 2, np.nan, 0], index=['a', 'b', 'c', 'd'])
    ms_ser = mpd.Series([np.nan, 2, np.nan, 0], index=['a', 'b', 'c', 'd'])
    values_ser_zero = pd.Series(np.zeros(4), index=['a', 'b', 'c', 'd'])
    ms_values_ser_zero = mpd.Series(np.zeros(4), index=['a', 'b', 'c', 'd'])
    assert ser.fillna(values_ser_zero).equals(ms_ser.fillna(ms_values_ser_zero).to_pandas())

    values_ser_nonzero = pd.Series(np.arange(4), index=['a', 'b', 'c', 'd'])
    ms_values_ser_nonzero = mpd.Series(np.arange(4), index=['a', 'b', 'c', 'd'])
    assert ser.fillna(values_ser_nonzero).equals(ms_ser.fillna(ms_values_ser_nonzero).to_pandas())


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_df_with_ser():
    """
    Test DataFrame.fillna() with a Series as an input value
    Description: tests df.fillna(ser)
    Expectation: same output as pandas df.fillna(ser)
    """
    df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                       [3, 4, np.nan, 1],
                       [np.nan, np.nan, np.nan, np.nan],
                       [np.nan, 3, np.nan, 4]],
                      columns=['a', 'b', 'c', 'd'])
    ms_df = mpd.DataFrame([[np.nan, 2, np.nan, 0],
                           [3, 4, np.nan, 1],
                           [np.nan, np.nan, np.nan, np.nan],
                           [np.nan, 3, np.nan, 4]],
                          columns=['a', 'b', 'c', 'd'])
    values_ser_zero = pd.Series(np.zeros(4), index=['a', 'b', 'c', 'd'])
    ms_values_ser_zero = mpd.Series(np.zeros(4), index=['a', 'b', 'c', 'd'])
    assert df.fillna(values_ser_zero).equals(ms_df.fillna(ms_values_ser_zero).to_pandas())

    values_ser_nonzero = pd.Series(np.arange(4), index=['a', 'b', 'c', 'd'])
    ms_values_ser_nonzero = mpd.Series(np.arange(4), index=['a', 'b', 'c', 'd'])
    assert df.fillna(values_ser_nonzero).equals(ms_df.fillna(ms_values_ser_nonzero).to_pandas())


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_fillna():
    """
    Test fillna
    Description: tests df.fillna
    Expectation: same output as pandas df.fillna
    """

    def test_fillna_fn(df):
        df = df.fillna(0)
        return df

    TESTUTIL.compare(test_fillna_fn, TESTUTIL.create_df_range_float)

# Noate: Although in documentation (https://pandas.pydata.org/docs/reference/api/pandas.Series.fillna.html)
#   the series.fillna does support dataframe as an input value, when we do so, the origin pandas will output
#   error "value" parameter must be a scalar, dict or Series, but you passed a "DataFrame".
# def test_ser_with_df():
#     '''
#         Test Series.fillna() with a DataFrame as an input value
#     '''
#     values_df_zero = pd.DataFrame(np.zeros(4), index=['a', 'b', 'c', 'd'])
#     ms_values_df_zero = mpd.DataFrame(np.zeros(4), index=['a', 'b', 'c', 'd'])
#     assert ser.fillna(values_df_zero).equals(ms_ser.fillna(ms_values_df_zero).to_pandas())
