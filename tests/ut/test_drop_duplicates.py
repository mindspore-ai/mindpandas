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
def test_drop_duplicates():
    """
    Test drop_duplicates
    Description: tests df.drop_duplicates
    Expectation: same output as pandas df.drop_duplicates
    """

    def test_drop_duplicates_all(df):
        df = df.drop_duplicates()
        return df

    def test_one_column_a(df):
        df = df.drop_duplicates(subset=['A'])
        return df

    def test_one_column_b(df):
        df = df.drop_duplicates(subset=['B'])
        return df

    def test_drop_duplicates_with_ignore_index(df):
        """See https://e.gitee.com/mind_spore/projects/67813/issues/list?issue=I4RDWF"""
        df = df.drop_duplicates(keep=False, ignore_index=True)
        return df

    def test_drop_duplicates_in_place(df):
        """See https://e.gitee.com/mind_spore/projects/67813/issues/list?issue=I4RDWF"""
        df.drop_duplicates(inplace=True)
        return df

    TESTUTIL.compare(test_drop_duplicates_all, create_fn=TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_one_column_a, create_fn=TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_one_column_b, create_fn=TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_drop_duplicates_with_ignore_index,
                     create_fn=TESTUTIL.create_df_duplicates)
    TESTUTIL.compare(test_drop_duplicates_in_place, create_fn=TESTUTIL.create_df_duplicates)
