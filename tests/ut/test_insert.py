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

import pandas as pd
import mindpandas as mpd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_insert():
    """
    Test insert
    Description: tests df.insert
    Expectation: same output as pandas df.insert
    """

    def test_insert_int(df):
        df.insert(loc=2, column="inserted", value=256)
        return df

    def test_insert_list(df):
        row, _ = df.shape
        df.insert(loc=2, column="inserted", value=range(row))
        # df.insert(loc=2, column="inserted", value=[1, 4, 2, 8, 5, 7])
        return df

    def test_insert_series(df):
        row, _ = df.shape
        df.insert(loc=2, column="inserted", value=pd.Series(range(row)))
        return df

    def test_adding_pd_series(df):
        pd_series = pd.Series([i for i in range(10, 16)])
        df.insert(0, 'newcol', pd_series, True)
        return df

    def test_adding_mpd_series(df):
        mpd_series = mpd.Series([i for i in range(10, 16)])
        df.insert(0, 'newcol', mpd_series, True)
        return df

    def test_repeated_adding_same_location(df):
        for _ in range(10):
            mpd_series = mpd.Series([i for i in range(10, 16)])
            df.insert(0, f"newcol_{_}", mpd_series, True)
        return df

    TESTUTIL.compare(test_insert_int, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_insert_list, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_insert_series, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_adding_pd_series, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_adding_mpd_series, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_repeated_adding_same_location, create_fn=TESTUTIL.create_df_small)
