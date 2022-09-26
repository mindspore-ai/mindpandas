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


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_apply(run_long=True):
    """
    Test apply
    Description: tests df.apply
    Expectation: same output as pandas df.apply
    """

    def test_numpy_universal(df):
        return df.apply(np.sqrt)

    def test_numpy_reduce_0(df):
        return df.apply(np.sum, axis=0)

    def test_numpy_reduce_1(df):
        return df.apply(np.sum, axis=1)

    def test_lambda_listlike(df):
        return df.apply(lambda _: [1, 2], axis=1)

    def test_result_type_expand(df):
        return df.apply(lambda _: [1, 2], axis=1, result_type="expand")

    def test_lambda_series_with_index(df):
        return df.apply(lambda _: pd.Series([1, 2], index=['foo', 'bar']), axis=1)

    def test_string_function(df):
        return df.apply("sum", axis=0)

    def test_list_function(df):
        return df.apply(["cumsum", np.sqrt], axis=0)

    def test_dict_function(df):
        return df.apply({'A': ["cumsum", np.sqrt],
                         'B': [np.sin, np.cos]}, axis=0)

    def test_column_function(df):
        def parse(x):
            return int(x + 1), int(x + 2)
        df[['key', 'edge_val']] = df.apply(parse, axis=1, result_type='expand')
        df = df[['key', 'edge_val']]
        return df

    TESTUTIL.compare(test_numpy_universal, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_numpy_reduce_0, create_fn=TESTUTIL.create_df_range)
    if run_long:
        TESTUTIL.compare(test_numpy_reduce_1, create_fn=TESTUTIL.create_df_range)
        TESTUTIL.compare(test_lambda_listlike, create_fn=TESTUTIL.create_df_range)
        TESTUTIL.compare(test_result_type_expand, create_fn=TESTUTIL.create_df_range)
        TESTUTIL.compare(test_lambda_series_with_index, create_fn=TESTUTIL.create_df_range)
        # for some reason dict_function fails with large table.
        # it works for functional test though
        TESTUTIL.compare(test_dict_function, create_fn=TESTUTIL.create_df_range)
        TESTUTIL.compare(test_list_function, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_string_function, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_column_function, create_fn=TESTUTIL.create_single_column_df)
