# Copyright 2023 Huawei Technologies Co., Ltd
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
def test_memory_usage():
    """
    Test memory_usage
    Description: tests df.memory_usage & ser.memory_usage
    Expectation: same output as pandas df.memory_usage & ser.memory_usage
    """

    def test_memory_usage_default(df):
        return df.memory_usage()

    def test_memory_usage_index(df):
        return df.memory_usage(index=False)

    def test_memory_usage_deep(df):
        return df.memory_usage(deep=True)

    TESTUTIL.compare(test_memory_usage_default, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_memory_usage_index, create_fn=TESTUTIL.create_df_mixed_dtypes)
    TESTUTIL.compare(test_memory_usage_deep, create_fn=TESTUTIL.create_df_mixed_dtypes)

    TESTUTIL.compare(test_memory_usage_default, create_fn=TESTUTIL.create_series_small)
    TESTUTIL.compare(test_memory_usage_index, create_fn=TESTUTIL.create_series_small)
    TESTUTIL.compare(test_memory_usage_deep, create_fn=TESTUTIL.create_series_small)
