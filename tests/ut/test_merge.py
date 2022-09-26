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

import pandas as pd
import mindpandas as mpd

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_merge():
    """
    Test merge
    Description: tests df.merge
    Expectation: same output as pandas df.merge
    """

    def test_merge_left(df):
        left, right = df
        return left.merge(right, how="left", left_on="lkey", right_on="rkey")

    def test_merge_inner(df):
        left, right = df
        return left.merge(right, how="inner", left_on="lkey", right_on="rkey")

    def test_merge_outer(df):
        left, right = df
        return left.merge(right, how="outer", left_on="lkey", right_on="rkey")

    def test_merge_issue1():
        data1 = {'key1': ['a', 'b', 'c', 'd'], 'key2': [
            'e', 'f', 'g', 'h'], 'key3': ['i', 'j', 'k', 'l']}
        data2 = {'key1': ['a', 'B', 'c', 'd'], 'key2': [
            'e', 'f', 'g', 'H'], 'key3': ['i', 'j', 'K', 'L']}
        index1 = ['k', 'l', 'm', 'n']
        index2 = ['p', 'q', 'u', 'v']
        df1 = pd.DataFrame(data1, index=index1)
        df2 = pd.DataFrame(data2, index=index2)
        mdf1 = mpd.DataFrame(data1, index=index1)
        mdf2 = mpd.DataFrame(data2, index=index2)
        assert df1.merge(df2, how='left').equals(mdf1.merge(mdf2, how='left').to_pandas())

    test_merge_issue1()
    TESTUTIL.compare(test_merge_left, create_fn=TESTUTIL.create_df_merge)
    TESTUTIL.compare(test_merge_inner, create_fn=TESTUTIL.create_df_merge)
    TESTUTIL.compare(test_merge_outer, create_fn=TESTUTIL.create_df_merge)
