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
def test_to_numpy():
    """
    Test to_numpy
    Description: tests df.to_numpy and Series.to_numpy
    Expectation: same output as pandas
    """

    def test_to_numpy_default(input_value):
        return input_value.to_numpy()

    def test_to_numpy_dtype(input_value):
        return input_value.to_numpy(dtype=object)

    TESTUTIL.compare(test_to_numpy_default, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_to_numpy_dtype, create_fn=TESTUTIL.create_df_array)
    TESTUTIL.compare(test_to_numpy_default, create_fn=TESTUTIL.create_series_range)
    TESTUTIL.compare(test_to_numpy_dtype, create_fn=TESTUTIL.create_series_range)