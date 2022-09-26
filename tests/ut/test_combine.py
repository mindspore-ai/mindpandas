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
def test_combine():
    """
    Test combine(df1, df2, combine_func)
    Description: tests df.combine
    Expectation: same output as pandas df.combine
    """

    def create_combine_test_df1_more_cols(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(500, 100))
        y = np.random.randint(100, size=(500, 50))
        df1 = module.DataFrame(x, columns=np.random.shuffle(list(range(x.shape[1]))))
        df2 = module.DataFrame(y, columns=np.random.shuffle(list(range(y.shape[1]))))
        return (df1, df2)

    def create_combine_test_df2_more_cols(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(500, 50))
        y = np.random.randint(100, size=(500, 100))
        df1 = module.DataFrame(x, columns=np.random.shuffle(list(range(x.shape[1]))))
        df2 = module.DataFrame(y, columns=np.random.shuffle(list(range(y.shape[1]))))
        return (df1, df2)

    def create_combine_test_equal_cols(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(500, 75))
        y = np.random.randint(100, size=(500, 75))
        df1 = module.DataFrame(x, columns=np.random.shuffle(list(range(x.shape[1]))))
        df2 = module.DataFrame(y, columns=np.random.shuffle(list(range(y.shape[1]))))
        return (df1, df2)

    def create_combine_test_nonintersecting_nan(module):
        df1 = module.DataFrame({'J': [-3, np.nan], 'K': [2, -1]})
        df2 = module.DataFrame({'K': [np.nan, -5], 'L': [8, np.nan]})
        return (df1, df2)

    def create_combine_test_typecast(module):
        df1 = module.DataFrame({'J': ['W', 'x'], 'K': ['y', 'Z']})
        df2 = module.DataFrame({'K': [2, -5], 'L': [8, -4]})
        return (df1, df2)

    def test_combine_fn(dfs):
        df1 = dfs[0]
        df2 = dfs[1]

        def take_smaller(ser1, ser2):
            return ser1 if ser1.sum() is ser2.sum() else ser2

        df = df1.combine(df2, take_smaller)
        return df

    def test_combine_fn_fill_overwrite(dfs):
        df1 = dfs[0]
        df2 = dfs[1]

        def take_smaller(ser1, ser2):
            return ser1 if ser1.sum() < ser2.sum() else ser2

        df = df1.combine(df2, take_smaller, fill_value=6, overwrite=False)
        return df

    TESTUTIL.compare(test_combine_fn, create_fn=create_combine_test_df1_more_cols)
    TESTUTIL.compare(test_combine_fn, create_fn=create_combine_test_df2_more_cols)
    TESTUTIL.compare(test_combine_fn, create_fn=create_combine_test_equal_cols)
    TESTUTIL.compare(test_combine_fn, create_fn=create_combine_test_typecast)
    TESTUTIL.compare(test_combine_fn_fill_overwrite,
                     create_fn=create_combine_test_nonintersecting_nan)
