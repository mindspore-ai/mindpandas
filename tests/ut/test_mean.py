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
def test_mean():
    """
    Test mean
    Description: tests df.mean
    Expectation: same output as pandas df.mean
    """

    def test_mean_default(df):
        df = df.mean()
        return df

    def test_mean_level_full(df):
        df = df.mean(axis=0, level='Full')
        return df

    def test_mean_level_partial(df):
        df = df.mean(axis=0, level='Partial')
        return df

    def test_mean_level_id(df):
        df = df.mean(axis=0, level='ID')
        return df

    def create_multindex_series(module):
        """ Create Multindex Series.
        """
        arrays = [
            ["block1", "block1", "block2", "block2", "block3", "block3", "block4", "block4"],
            ["sub1", "sub2", "sub1", "sub2", "sub1", "sub2", "sub1", "sub2"],
        ]
        index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["index_one", "index_two"])
        series = module.Series(np.array([i for i in range(1, 9)]), index=index)
        return series

    def test_mean_multindex_series_level(series):
        """ Test multi-index series mean() with parameter <level>.
        """
        return series.mean(level="index_one")

    def test_mean_multindex_series_level2(series):
        """ Test multi-index series mean() with parameter <level>.
        """
        return series.mean(level="index_two")

    def test_mean_multindex_series_level3(series):
        """ Test multi-index series mean() with parameter <level>.
        """
        return series.mean(level=0)

    def test_err_mean_series_numeric_only(series):
        """ mean() is not implemented for Series in Pandas, test Error message.
        """
        return series.mean(numeric_only=True)

    def test_err_mean_df_string_numeric_only_is_false(df):
        """ mean(numeric_only = False) in Dataframe with string value
        will raise Error. Test Error message.
        """
        return df.mean(numeric_only=False)

    def test_err_mean_df_string_numeric_only_is_string(df):
        """ mean(numeric_only = other) is same with mean(numeric_only = False)
        in Dataframe with string value will raise Error. Test Error message.
        """
        return df.mean(numeric_only="Test")

    def test_mean_axis(df):
        return df.mean(axis=1)

    TESTUTIL.compare(test_mean_default, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_mean_default, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_mean_axis, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_mean_axis, create_fn=TESTUTIL.create_df_mixed_dtypes)

    TESTUTIL.compare(test_mean_level_full, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_mean_level_partial, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_mean_level_id, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_mean_multindex_series_level, create_fn=create_multindex_series)
    TESTUTIL.compare(test_mean_multindex_series_level2, create_fn=create_multindex_series)
    TESTUTIL.compare(test_mean_multindex_series_level3, create_fn=create_multindex_series)

    # tests mindspore.pandas has the same Error message as Pandas
    TESTUTIL.run_compare_error(test_err_mean_series_numeric_only,
                               NotImplementedError, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.run_compare_error(test_err_mean_series_numeric_only,
                               NotImplementedError, create_fn=create_multindex_series)
    TESTUTIL.run_compare_error(test_err_mean_df_string_numeric_only_is_false,
                               TypeError, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.run_compare_error(test_err_mean_df_string_numeric_only_is_false,
                               TypeError, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.run_compare_error(test_err_mean_df_string_numeric_only_is_string,
                               TypeError, create_fn=TESTUTIL.create_df_mixed_dtypes)
