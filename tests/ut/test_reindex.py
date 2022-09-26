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
def test_reindex():
    """
    Test reindex
    Description: tests df.reindex
    Expectation: same output as pandas df.reindex
    """

    new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10']

    def test_reindex_for_new_index(df):
        df = df.reindex(new_index)
        return df

    def test_reindex_for_fill_zero(df):
        df = df.reindex(new_index, fill_value=0)
        return df

    def test_reindex_for_fill_missing(df):
        df = df.reindex(new_index, fill_value='missing')
        return df

    def test_reindex_for_columns(df):
        df = df.reindex(columns=['http_status', 'user_agent'])
        return df

    def test_reindex_for_axis(df):
        df = df.reindex(['http_status', 'user_agent'], axis=1)
        return df

    def create_df_reindex(module):
        data = {1000: [10, 20, 30, 40, 50, 60], 2000: [100, 200, 300, 400, 500, 600]}
        index = [1, 2, 3, 4, 5, 6]
        return module, module.DataFrame(data, index)

    def test_reindex_tolerance(df_mod):
        module, df = df_mod
        ser = module.Series([1, 2, 1])
        df = df.reindex(index=[1, 5, 7], method='pad', tolerance=ser)
        return df

    TESTUTIL.compare(test_reindex_for_new_index, create_fn=TESTUTIL.create_df_reindex)
    TESTUTIL.compare(test_reindex_for_fill_zero, create_fn=TESTUTIL.create_df_reindex)
    TESTUTIL.compare(test_reindex_for_fill_missing, create_fn=TESTUTIL.create_df_reindex)
    TESTUTIL.compare(test_reindex_for_columns, create_fn=TESTUTIL.create_df_reindex)
    TESTUTIL.compare(test_reindex_for_axis, create_fn=TESTUTIL.create_df_reindex)
    TESTUTIL.compare(test_reindex_tolerance, create_fn=create_df_reindex)
