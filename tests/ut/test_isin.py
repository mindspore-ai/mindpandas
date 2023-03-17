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

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_isin():
    """
    Test isin
    Description: tests df.isin
    Expectation: same output as pandas df.isin
    """

    def create_df_isin(module):
        data = {'num_legs': [2, 4], 'num_wings': [2, 0]}
        index = ['falcon', 'dog']
        return module, module.DataFrame(data, index)

    def test_isin_list(df):
        df = df.isin([7, 8, 9])
        return df

    def test_isin_series(df_mod):
        module, df = df_mod
        df = df.isin(module.Series([0, 2], index=['dog', 'falcon']))
        return df

    TESTUTIL.compare(test_isin_list, create_fn=TESTUTIL.create_df_range)
    TESTUTIL.compare(test_isin_series, create_fn=create_df_isin)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_series_isin():
    """
    Test isin
    Description: tests Series.isin()
    Expectation: same output as pandas.Series.isin()
    """
    def create_series_isin(module):
        return module.Series(np.random.randint(1, 6, 100))

    def create_series_str_isin(module):
        data = ['a', 'b', 'c', 'd', 'e']
        return module.Series(data)

    def test_series_isin_list(ser):
        result = ser.isin([1, 2, 3])
        return result

    def test_series_isin_dict(ser):
        values = {'a': 'a', 'b': '1', '2': 'c'}
        result = ser.isin(values)
        return result

    TESTUTIL.compare(test_series_isin_list, create_fn=create_series_isin)
    TESTUTIL.compare(test_series_isin_dict, create_fn=create_series_str_isin)
