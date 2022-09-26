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
def test_filter_with_series():
    """
    Test filter with series
    Description: tests df[ser]
    Expectation: same output as pandas df[ser]
    """

    def result(df):
        return df

    def df_filter(module):
        df = module.DataFrame({'id': [1, 2, 3, 4], 'docid': [101, 102, 110, 123]})
        ser = module.Series([True, False, False, True])
        return df[ser]

    TESTUTIL.compare(result, df_filter)
