# Copyright 2023-2023 Huawei Technologies Co., Ltd
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
def test_abs():
    """
    Test abs
    Description: tests df.abs
    Expectation: same output as pandas.DataFrame.abs
    """

    def test_abs_pandas(df):
        df = df.abs()
        return df

    def test_abs_python(df):
        df = abs(df)
        return df

    TESTUTIL.compare(test_abs_pandas, create_fn=TESTUTIL.create_df_empty)
    TESTUTIL.compare(test_abs_pandas, create_fn=TESTUTIL.create_df_empty_with_columns)
    TESTUTIL.compare(test_abs_pandas, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_abs_pandas, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_abs_pandas, create_fn=TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_abs_pandas, create_fn=TESTUTIL.create_series_complex)

    TESTUTIL.compare(test_abs_python, create_fn=TESTUTIL.create_df_empty)
    TESTUTIL.compare(test_abs_python, create_fn=TESTUTIL.create_df_empty_with_columns)
    TESTUTIL.compare(test_abs_python, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_abs_python, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_abs_python, create_fn=TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_abs_python, create_fn=TESTUTIL.create_series_complex)
