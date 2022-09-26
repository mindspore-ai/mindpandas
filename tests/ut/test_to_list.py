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
def test_tolist():
    """
    Test tolist
    Description: tests df.tolist
    Expectation: same output as pandas df.tolist
    """

    def test_tolist_fn(df):
        return df.tolist()

    TESTUTIL.compare(test_tolist_fn, create_fn=TESTUTIL.create_series_range)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_to_list():
    """
    Test to_list
    Description: tests df.to_list
    Expectation: same output as pandas df.to_list
    """

    def test_to_list_fn(df):
        return df.to_list()

    TESTUTIL.compare(test_to_list_fn, create_fn=TESTUTIL.create_series_range)
