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
def test_series_value_counts():
    """
    Test Series_value_counts
    Description: tests Series.value_counts
    Expectation: same output as pandas Series.value_counts
    """

    def test_sort_is_false(ser):
        return ser.value_counts(sort=False)

    def test_default_param(ser):
        return ser.value_counts()

    def test_normalize(ser):
        return ser.value_counts(normalize=True)

    TESTUTIL.compare(test_sort_is_false, create_fn=TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_default_param, create_fn=TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_normalize, create_fn=TESTUTIL.create_series_dup)
