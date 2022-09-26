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
import numpy as np

import pandas as pd

from util import TESTUTIL


pytest.mark.usefixtures("set_mode", "set_shape")
def test_reset_index():
    """
    Test reset_index
    Description: tests df.reset_index
    Expectation: same output as pandas df.reset_index
    """

    def create_df_multindex_singlelevel(module):
        # multindex
        index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
                                           ('bird', 'parrot'),
                                           ('mammal', 'lion'),
                                           ('mammal', 'monkey')],
                                          names=['class', 'name'])
        # singlelevel
        columns = pd.MultiIndex.from_tuples([['speed'], ['species']])
        df = module.DataFrame([(389.0, 'fly'),
                               (24.0, 'fly'),
                               (80.5, 'run'),
                               (np.nan, 'jump')],
                              index=index,
                              columns=columns)

        return df

    def create_df_multindex_multilevel(module):
        # multindex
        index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
                                           ('bird', 'parrot'),
                                           ('mammal', 'lion'),
                                           ('mammal', 'monkey')],
                                          names=['class', 'name'])
        # multilevel
        columns = pd.MultiIndex.from_tuples([('speed', 'max'),
                                             ('species', 'type')])
        df = module.DataFrame([(389.0, 'fly'),
                               (24.0, 'fly'),
                               (80.5, 'run'),
                               (np.nan, 'jump')],
                              index=index,
                              columns=columns)

        return df

    def test_reset_index_default(df):
        return df.reset_index()

    def test_reset_index_drop_false(df):
        return df.reset_index(drop=True)

    # New reset_index Tests under multindex Datafame
    # Dataframe with multindex and singlelevel
    def test_reset_index_multindex_singlelevel(df):
        return df.reset_index()

    def test_reset_index_multindex_singlelevel_inplace(df):
        df.reset_index(inplace=True)
        return df

    def test_reset_index_multindex_singlelevel_drop(df):
        return df.reset_index(drop=True)

    def test_reset_index_multindex_singlelevel_all(df):
        df.reset_index(inplace=True, drop=True)
        return df

    # Dataframe with multindex and multiLevel
    def test_reset_index_multindex_multilevel(df):
        return df.reset_index()

    def test_reset_index_multindex_multilevel_level(df):
        return df.reset_index(level='class')

    def test_reset_index_multindex_multilevel_col_level(df):
        return df.reset_index(col_level=1)

    def test_reset_index_multindex_multilevel_col_all_1(df):
        df.reset_index(inplace=True, level='class', col_level=1, col_fill='new_test')
        return df

    def test_reset_index_multindex_multilevel_col_all_2(df):
        return df.reset_index(drop=True, level='class')

    TESTUTIL.compare(test_reset_index_default, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_reset_index_drop_false, create_fn=TESTUTIL.create_df_small)
    # New Tests
    TESTUTIL.compare(test_reset_index_multindex_singlelevel,
                     create_fn=create_df_multindex_singlelevel)
    TESTUTIL.compare(test_reset_index_multindex_singlelevel_inplace,
                     create_fn=create_df_multindex_singlelevel)
    TESTUTIL.compare(test_reset_index_multindex_singlelevel_drop,
                     create_fn=create_df_multindex_singlelevel)
    TESTUTIL.compare(test_reset_index_multindex_singlelevel_all,
                     create_fn=create_df_multindex_singlelevel)
    TESTUTIL.compare(test_reset_index_multindex_multilevel,
                     create_fn=create_df_multindex_multilevel)
    TESTUTIL.compare(test_reset_index_multindex_multilevel_level,
                     create_fn=create_df_multindex_multilevel)
    TESTUTIL.compare(test_reset_index_multindex_multilevel_col_level,
                     create_fn=create_df_multindex_multilevel)
    TESTUTIL.compare(test_reset_index_multindex_multilevel_col_all_1,
                     create_fn=create_df_multindex_multilevel)
    TESTUTIL.compare(test_reset_index_multindex_multilevel_col_all_2,
                     create_fn=create_df_multindex_multilevel)
