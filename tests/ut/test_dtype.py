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
def test_dtype():
    """
    Test dtype
    Description: tests df.dtype
    Expectation: same output as pandas df.dtype
    """

    def test_dtypes(df):
        return df.dtypes

    def test_dtype_fn(ser):
        return ser.dtype

    TESTUTIL.compare(test_dtypes)
    TESTUTIL.compare(test_dtypes, create_fn=TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_dtypes, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_dtype_fn, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_dtypes, create_fn=TESTUTIL.create_series_unique)
    TESTUTIL.compare(test_dtypes, create_fn=TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_dtypes, create_fn=TESTUTIL.create_series_bool)
    TESTUTIL.compare(test_dtypes, create_fn=TESTUTIL.create_series_nan)
