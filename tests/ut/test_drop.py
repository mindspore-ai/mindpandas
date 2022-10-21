# Copyright 2022 Huawei Technologies Co., Ltd
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
def test_dataframe_drop():
    """
    Test drop
    Description: tests DataFrame.drop
    Expectation: same output as pandas DataFrame.drop
    """

    def test_drop_index(df):
        df = df.drop(index=[1, 50, 99])
        return df

    def test_drop_index_by_labels(df):
        df = df.drop(labels=[1, 50, 99], axis=0)
        return df

    def test_drop_column(df):
        df = df.drop(columns=["A", "C", "D"])
        return df

    def test_drop_column_by_labels(df):
        df = df.drop(labels=["A", "C", "D"], axis=1)
        return df

    def test_drop_inplace(df):
        df.drop(columns=["A", "C", "D"], inplace=True)
        return df

    TESTUTIL.compare(test_drop_index, TESTUTIL.create_df_range)
    TESTUTIL.compare(test_drop_index_by_labels, TESTUTIL.create_df_range)
    TESTUTIL.compare(test_drop_column, TESTUTIL.create_df_range)
    TESTUTIL.compare(test_drop_column_by_labels, TESTUTIL.create_df_range)
    TESTUTIL.compare(test_drop_inplace, TESTUTIL.create_df_range)
