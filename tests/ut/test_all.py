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
def test_all():
    """
    Test all
    Description: tests df.all
    Expectation: same output as pandas df.all
    """

    def test_all_fn(df):
        df = df.all()
        return df

    def test_all_axis_none(df):
        df = df.all(axis=None)
        return df

    def test_all_bool_only(df):
        df = df.all(bool_only=True)
        return df

    TESTUTIL.compare(test_all_fn, TESTUTIL.create_df_bool)
    TESTUTIL.compare(test_all_axis_none, TESTUTIL.create_df_bool)
    TESTUTIL.compare(test_all_bool_only, TESTUTIL.create_df_bool_and_str)
