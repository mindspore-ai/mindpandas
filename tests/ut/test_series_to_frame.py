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
def test_series_to_frame():
    """
    Test Series_to_frame
    Description: tests ser.to_frame
    Expectation: same output as pandas ser.to_frame
    """

    def create_input_series_without_name(module):
        ser = module.Series(["a", "b", "c", "d"])
        return ser

    def create_input_series_with_name(module):
        ser = module.Series(["a", "b", "c", "d"], name="value")
        return ser

    def test_to_frame_change_name(ser):
        df = ser.to_frame(name="note")
        return df

    def test_to_frame_default_name(ser):
        df = ser.to_frame()
        return df

    TESTUTIL.compare(test_to_frame_change_name, create_input_series_without_name)
    TESTUTIL.compare(test_to_frame_change_name, create_input_series_with_name)
    TESTUTIL.compare(test_to_frame_default_name, create_input_series_without_name)
    TESTUTIL.compare(test_to_frame_default_name, create_input_series_with_name)
