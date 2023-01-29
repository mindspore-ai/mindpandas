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
def test_count():
    """
    Test count
    Description: tests df.count
    Expectation: same output as pandas.DataFrame.count
    """

    def test_count_default(df):
        return df.count()

    def test_count_axis_1(df):
        return df.count(axis=1)

    def test_count_level(df):
        return df.count(level='Full')

    def test_count_numeric_only_is_true(df):
        return df.count(numeric_only=True)


    TESTUTIL.compare(test_count_default, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_count_default, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_count_axis_1, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_count_axis_1, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_count_level, create_fn=TESTUTIL.create_hierarchical_df)
    TESTUTIL.compare(test_count_numeric_only_is_true, create_fn=TESTUTIL.create_df_mixed_dtypes)
