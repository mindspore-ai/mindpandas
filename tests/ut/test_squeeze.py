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
def test_squeeze():
    """
    Test squeeze
    Description: tests df.squeeze
    Expectation: same output as pandas df.squeeze
    """

    def test_squeeze_fn(df):
        df = df.squeeze()
        return df

    def test_squeeze_rows(df):
        df = df.squeeze('rows')
        return df

    def test_squeeze_columns(df):
        df = df.squeeze('columns')
        return df

    TESTUTIL.compare(test_squeeze_fn, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_squeeze_rows, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_squeeze_columns, create_fn=TESTUTIL.create_df_small)

    TESTUTIL.compare(test_squeeze_fn, create_fn=TESTUTIL.create_single_column_df)
    TESTUTIL.compare(test_squeeze_rows, create_fn=TESTUTIL.create_single_column_df)
    TESTUTIL.compare(test_squeeze_columns, create_fn=TESTUTIL.create_single_column_df)

    TESTUTIL.compare(test_squeeze_fn, create_fn=TESTUTIL.create_single_row_df)
    TESTUTIL.compare(test_squeeze_rows, create_fn=TESTUTIL.create_single_row_df)
    TESTUTIL.compare(test_squeeze_columns, create_fn=TESTUTIL.create_single_row_df)

    TESTUTIL.compare(test_squeeze_fn, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_squeeze_rows, create_fn=TESTUTIL.create_series_range)

    TESTUTIL.compare(test_squeeze_fn, create_fn=TESTUTIL.create_series_unique)
    TESTUTIL.compare(test_squeeze_rows, create_fn=TESTUTIL.create_series_unique)
