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
def test_selection_projection():
    """
    Test selection projection
    Description: tests df with selection projection
    Expectation: same output as pandas
    """

    def create_input_dataframe(module):
        df = module.DataFrame({'id': [1, 4, 5, 9], 'docid': [100, 101, 112, 123]})
        return df

    def test_project_select_1(df):
        # Return a Series
        return df.docid[df.id > 1]

    def test_project_select_2(df):
        # Return a DataFrame
        return df[['docid']][df['id'] <= 6]

    TESTUTIL.compare(test_project_select_1, create_input_dataframe)
    TESTUTIL.compare(test_project_select_2, create_input_dataframe)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_selection_on_join():
    """
    Test selection on join
    Description: tests df selection on join
    Expectation: same output as pandas
    """

    def create_input_dataframe(module):
        df1 = module.DataFrame(
            {'col11': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             'col12': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
             'col13': [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
             'col14': [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
             'col15': [81, 82, 83, 84, 85, 86, 87, 88, 89, 90]}
        )
        df2 = module.DataFrame(
            {'col21': [2, 4, 6, 8, 10],
             'col22': [0, 23, 27, 25, 21],
             'col23': [42, 0, 43, 0, 50],
             'col24': [68, 64, 0, 0, 0],
             'col25': [0, 0, 0, 0, 89]}
        )
        return (df1, df2)

    def test_project_inner_join(df):
        df1 = df[0]
        df2 = df[1]
        res1 = df1.merge(df2, how='inner', left_on='col11', right_on='col21')
        res2 = res1[res1.col12 >= 22][['col13', 'col22']]
        return res2

    def test_project_left_join_1(df):
        df1 = df[0]
        df2 = df[1]
        res1 = df1.merge(df2, how='left', left_on='col11', right_on='col21')
        res2 = res1[res1.col12 >= 22][['col13', 'col22']]
        return res2

    def test_project_left_join_2(df):
        # No rewrite in push_project_under_join
        # Future: convert left join to inner join then push down the pred under the inner join
        df1 = df[0]
        df2 = df[1]
        res1 = df1.merge(df2, how='left', left_on='col11', right_on='col21')
        res2 = res1[res1.col22 >= 22][['col13', 'col22']]
        return res2

    def test_project_right_join(df):
        df1 = df[0]
        df2 = df[1]
        res1 = df2.merge(df1, how='right', left_on='col21', right_on='col11')
        res2 = res1[res1.col12 >= 22][['col13', 'col22']]
        return res2

    TESTUTIL.compare(test_project_inner_join, create_input_dataframe, ignore_index=True)
    TESTUTIL.compare(test_project_left_join_1, create_input_dataframe, ignore_index=True)
    TESTUTIL.compare(test_project_left_join_2, create_input_dataframe, ignore_index=True)
    TESTUTIL.compare(test_project_right_join, create_input_dataframe,
                     ignore_index=True, compare_pd_ms=False)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_selection_on_groupby():
    """
    Test selection on groupby
    Description: tests df selection on groupby
    Expectation: same output as pandas
    """

    def create_input_dataframe(module):
        df1 = module.DataFrame({'col1': [3, 1, 4, 4, 1, 2, 3, 2, 1, 4],
                                'col2': [21, 21, 22, 22, 22, 21, 30, 21, 21, 30],
                                'col3': [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                                'col4': [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                                'col5': [10, 10, 10, 10, 10, 10, 11, 10, 10, 10]})
        return df1

    def test_select_on_sum(df):
        res1 = df.groupby(by='col1', as_index=False).sum()
        res2 = res1[res1.col1 < 4]
        return res2

    def test_select_on_count(df):
        res1 = df.groupby(by='col1', as_index=False).count()
        res2 = res1[res1.col1 > 1]['col3']
        return res2

    def test_select_on_size(df):
        res1 = df.groupby(by='col1', as_index=False).size()
        res2 = res1[res1.col1 != 3]
        return res2

    def test_select_on_min(df):
        res1 = df.groupby(by='col2', as_index=False).min()
        res2 = res1[res1.col2 == 30]
        return res2

    def test_select_on_max(df):
        res1 = df.groupby(by='col2', as_index=False).max()
        res2 = res1[res1.col2 >= 21][['col1', 'col3', 'col5']]
        return res2

    TESTUTIL.compare(test_select_on_sum, create_input_dataframe, ignore_index=True)
    TESTUTIL.compare(test_select_on_count, create_input_dataframe, ignore_index=True)
    TESTUTIL.compare(test_select_on_size, create_input_dataframe, ignore_index=True)
    TESTUTIL.compare(test_select_on_min, create_input_dataframe, ignore_index=True)
    TESTUTIL.compare(test_select_on_max, create_input_dataframe, ignore_index=True)
