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
def test_concat():
    """
    Test concat
    Description: tests df.concat and Series.concat
    Expectation: same output as pandas
    """

    def test_concat_axis0(module):
        df1 = TESTUTIL.default_create_fn(module)
        df2 = TESTUTIL.default_create_fn(module)
        df_list = [df1, df2]
        return module.concat(df_list, axis=0)

    def test_concat_axis1(module):
        df1 = TESTUTIL.default_create_fn(module)
        df2 = TESTUTIL.default_create_fn(module)
        df_list = [df1, df2]
        return module.concat(df_list, axis=1)

    def test_series_concat_axis0(module):
        ser1 = TESTUTIL.create_series_large(module)
        ser2 = TESTUTIL.create_series_large(module)
        ser_list = [ser1, ser2]
        return module.concat(ser_list, axis=0)

    def test_series_concat_axis1(module):
        ser1 = TESTUTIL.create_series_large(module)
        ser2 = TESTUTIL.create_series_large(module)
        ser_list = [ser1, ser2]
        return module.concat(ser_list, axis=1)

    def test_noop(df):
        return df

    TESTUTIL.compare(test_concat_axis0, test_noop)
    TESTUTIL.compare(test_concat_axis1, test_noop)
    TESTUTIL.compare(test_series_concat_axis0, test_noop)
    TESTUTIL.compare(test_series_concat_axis1, test_noop)
