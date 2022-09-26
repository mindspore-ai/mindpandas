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
def test_set_index():
    """
    Test set_index
    Description: tests df.set_index
    Expectation: same output as pandas df.set_index
    """

    def test_set_index_from_columns(df):
        df = df.set_index('month')
        return df

    def test_set_index_all_columns(df):
        df = df.set_index(['month', 'year', 'sale'])
        return df

    def test_set_index_inplace(df):
        df.set_index('month', inplace=True)

    def test_set_index_from_unsorted_columns(df):
        df = df.set_index(['year', 'month'])
        return df

    TESTUTIL.compare(test_set_index_from_columns, create_fn=TESTUTIL.create_df_setindex)
    TESTUTIL.compare(test_set_index_all_columns, create_fn=TESTUTIL.create_df_setindex)
    TESTUTIL.compare(test_set_index_inplace, create_fn=TESTUTIL.create_df_setindex)
    TESTUTIL.compare(test_set_index_from_unsorted_columns, create_fn=TESTUTIL.create_df_setindex)
