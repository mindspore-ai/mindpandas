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
def test_copy():
    """
    Test copy
    Description: tests df.copy
    Expectation: same output as pandas df.copy
    """

    def test_copy_deep(df):
        df_copy = df.copy()
        df_copy[0][0] = 1000
        return df

    def test_copy_deep_str(df):
        df_copy = df.copy()
        df_copy["columns1"]["index1"] = 1000
        return df

    def test_copy_shallow(df):
        df_copy = df.copy(False)
        df_copy[0][0] = 1000
        return df

    def test_copy_series_deep(ser):
        ser_copy = ser.copy()
        ser_copy[0] = 0
        return ser

    def test_copy_series_shallow(ser):
        ser_copy = ser.copy(False)
        ser_copy[0] = 0
        return ser

    TESTUTIL.compare(test_copy_deep, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_copy_deep_str, create_fn=TESTUTIL.create_df_with_columns_and_index)
    TESTUTIL.compare(test_copy_shallow, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_copy_series_deep, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_copy_series_shallow, create_fn=TESTUTIL.create_series_range)
