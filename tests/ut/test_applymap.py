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
def test_applymap():
    """
    Test applymap
    Description: tests df.applymap
    Expectation: same output as pandas df.applymap
    """

    def test_func(df):
        df = df.applymap(lambda x: len(str(x)))
        return df

    def test_na_ignore(df):
        df = df.applymap(lambda x: len(str(x)), na_action='ignore')
        return df

    def test_square(df):
        df = df.applymap(lambda x: x ** 2)
        return df

    TESTUTIL.compare(test_func, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_na_ignore, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_square, create_fn=TESTUTIL.create_df_small)
