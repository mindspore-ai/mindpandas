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
def test_dropna():
    """
    Test dropna
    Description: tests df.dropna
    Expectation: same output as pandas df.dropna
    """

    def test_dropna_fn(df):
        df = df.dropna()
        return df

    def test_dropna_axis1_fn(df):
        df = df.dropna(axis=1)
        return df

    def test_dropna_how_fn(df):
        df = df.dropna(how='all')
        return df

    TESTUTIL.compare(test_dropna_fn, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_dropna_axis1_fn, TESTUTIL.create_df_range_float)
    TESTUTIL.compare(test_dropna_how_fn, TESTUTIL.create_df_range_float)
