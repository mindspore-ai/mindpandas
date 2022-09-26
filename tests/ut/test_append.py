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
def test_append():
    """
    Test DataFrame append
    Description: tests df.append
    Expectation: same output as pandas df.append
    """

    def test_append_fn(df):
        return df.append(df)

    def test_append_ignore_index(df):
        return df.append(df, ignore_index=True)

    def test_append_sort(df):
        return df.append(df, sort=True)

    TESTUTIL.compare(test_append_fn, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_append_ignore_index, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_append_sort, TESTUTIL.create_df_range_float)
