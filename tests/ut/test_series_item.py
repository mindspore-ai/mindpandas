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
def test_series_item():
    """
    Test Series_item
    Description: tests Series.item()
    Expectation: same output as pandas.Series.item()
    """

    def create_series_integer(module):
        ser = module.Series([1])
        return ser

    def create_series_string(module):
        ser = module.Series(["a"], name="value")
        return ser

    def test_series_item_normal(ser):
        return ser.item()

    def test_series_item_exception(ser):
        with pytest.raises(ValueError):
            return ser.item()

    TESTUTIL.compare(test_series_item_normal, create_series_integer)
    TESTUTIL.compare(test_series_item_normal, create_series_string)
    TESTUTIL.compare(test_series_item_exception, TESTUTIL.create_series_small)
    TESTUTIL.compare(test_series_item_exception, TESTUTIL.create_series_small)
