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
def test_std():
    """
    Test Dataframe and Series std
    Description: tests df.std and Series.std
    Expectation: same output as pandas
    """

    def test_std_fn_axis0(df):
        df = df.std(axis=0)
        return df

    def test_std_fn_axis1(df):
        df = df.std(axis=1)
        return df

    def test_std_fn_level(df):
        df = df.std(level=0)
        return df

    def test_std_fn_ddof(df):
        df = df.std(ddof=0)
        return df

    def test_err_std_series_numeric_only(series):
        """ std() is not implemented for Series in Pandas, test Error message.
        """
        return series.std(numeric_only=True)

    def test_err_std_df_string_numeric_only_is_false(df):
        """ std(numeric_only = False) in Dataframe with string value
        will raise Error. Test Error message.
        """
        return df.std(numeric_only=False)

    def test_std_multindex_series_level(series):
        """ Test multi-index series mean() with parameter <level>.
        """
        return series.std(level="blooded")

    TESTUTIL.compare(test_std_fn_axis0, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_std_fn_axis1, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_std_fn_level, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_std_fn_ddof, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_std_fn_axis0, TESTUTIL.create_series_range)
    TESTUTIL.compare(test_std_fn_level, TESTUTIL.create_series_range)
    TESTUTIL.compare(test_std_fn_ddof, TESTUTIL.create_series_range)
    TESTUTIL.compare(test_std_multindex_series_level, TESTUTIL.create_hierarchical_series)
    TESTUTIL.run_compare_error(test_err_std_series_numeric_only,
                               NotImplementedError, TESTUTIL.create_series_range)
    TESTUTIL.run_compare_error(test_err_std_series_numeric_only,
                               NotImplementedError, TESTUTIL.create_hierarchical_series)
    TESTUTIL.run_compare_error(test_err_std_df_string_numeric_only_is_false,
                               TypeError, TESTUTIL.create_hierarchical_df)
