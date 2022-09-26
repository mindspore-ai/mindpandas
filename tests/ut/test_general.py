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
import numpy as np
import pytest

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_date_range():
    """
    Test data_range
    """

    def create_test_date_range_1(module):
        df = module.date_range(start='1/1/2022', end='1/31/2022')
        return df

    def create_test_date_range_2(module):
        df = module.date_range(start='1/1/2022', periods=64)
        return df

    def create_test_date_range_3(module):
        df = module.date_range(end='1/31/2022', periods=64)
        return df

    def create_test_date_range_4(module):
        df = module.date_range(start='1/1/2022', end='1/31/2022', periods=64)
        return df

    def create_test_date_range_5(module):
        df = module.date_range(start='1/1/2022', periods=32, freq='3M')
        return df

    def create_test_date_range_6(module):
        df = module.date_range(end='1/31/2022', periods=32, freq='3M')
        return df

    def create_test_date_range_7(module):
        df = module.date_range(start='1/1/2022', end='12/31/2022', freq='W')
        return df

    def create_test_date_range_8(module):
        df = module.date_range(start='1/1/2022', periods=32, tz='Asia/Tokyo')
        return df

    def test_data_range_fn(df):
        return df

    TESTUTIL.compare(test_data_range_fn, create_test_date_range_1)
    TESTUTIL.compare(test_data_range_fn, create_test_date_range_2)
    TESTUTIL.compare(test_data_range_fn, create_test_date_range_3)
    TESTUTIL.compare(test_data_range_fn, create_test_date_range_4)
    TESTUTIL.compare(test_data_range_fn, create_test_date_range_5)
    TESTUTIL.compare(test_data_range_fn, create_test_date_range_6)
    TESTUTIL.compare(test_data_range_fn, create_test_date_range_7)
    TESTUTIL.compare(test_data_range_fn, create_test_date_range_8)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_pivot_table():
    """
    Test pivot_table
    """

    def create_pivot_table_test_1(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(100, 50))
        df = module.DataFrame(x, columns=[chr(y) for y in range(65, 115)])
        df = module.pivot_table(df, index='A')
        return df

    def create_pivot_table_test_2(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(100, 50))
        df = module.DataFrame(x, columns=[chr(y) for y in range(65, 115)])
        df = module.pivot_table(df, values=['E'], index=['A'], columns=['C'])
        return df

    def create_pivot_table_test_3(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(100, 50))
        df = module.DataFrame(x, columns=[chr(y) for y in range(65, 115)])
        df = module.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], fill_value=-5)
        return df

    def create_pivot_table_test_4(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(100, 50))
        df = module.DataFrame(x, columns=[chr(y) for y in range(65, 115)])
        df = module.pivot_table(df, values=['D', 'E'], index=['A', 'C'], aggfunc=np.sum, dropna=False)
        return df

    def create_pivot_table_test_5(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(100, 50))
        df = module.DataFrame(x, columns=[chr(y) for y in range(65, 115)])
        df = module.pivot_table(df, columns=['C'], margins=True, margins_name='All')
        return df

    def create_pivot_table_test_6(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(100, 50))
        df = module.DataFrame(x, columns=[chr(y) for y in range(65, 115)])
        df = module.pivot_table(df,
                                values=['D', 'E'],
                                index=['A', 'C'],
                                aggfunc={'D': np.mean,
                                         'E': [min, max, np.mean]})
        return df

    def create_pivot_table_test_7(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(100, 50))
        df = module.DataFrame(x, columns=[chr(y) for y in range(65, 115)])
        df = module.pivot_table(df,
                                values=['D', 'E'],
                                index=['A'],
                                columns=['C'],
                                aggfunc=[min, max, np.mean],
                                fill_value=-5,
                                margins=True,
                                dropna=False,
                                margins_name='All',
                                observed=False,
                                sort=True)
        return df

    def test_pivot_table_fn(df):
        return df

    TESTUTIL.compare(test_pivot_table_fn, create_pivot_table_test_1)
    TESTUTIL.compare(test_pivot_table_fn, create_pivot_table_test_2)
    TESTUTIL.compare(test_pivot_table_fn, create_pivot_table_test_3)
    TESTUTIL.compare(test_pivot_table_fn, create_pivot_table_test_4)
    TESTUTIL.compare(test_pivot_table_fn, create_pivot_table_test_5)
    TESTUTIL.compare(test_pivot_table_fn, create_pivot_table_test_6)
    TESTUTIL.compare(test_pivot_table_fn, create_pivot_table_test_7)


if __name__ == "__main__":
    test_date_range()
    test_pivot_table()
