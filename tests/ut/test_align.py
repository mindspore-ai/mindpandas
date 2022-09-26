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
def test_align():
    """
    Test align
    Description: tests df.align
    Expectation: same output as pandas df.align
    """

    def create_align_test_1(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(500, 100))
        y = np.random.randint(100, size=(500, 50))
        df1 = module.DataFrame(x, columns=np.random.shuffle(list(range(x.shape[1]))))
        df2 = module.DataFrame(y, columns=np.random.shuffle(list(range(y.shape[1]))))
        return (df1, df2)

    def create_align_test_2(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(500, 100))
        y = np.random.randint(100, size=(100))
        df1 = module.DataFrame(x)
        df2 = module.Series(y)
        return (df1, df2)

    def create_align_test_3(module):
        np.random.seed(100)
        x = np.random.randint(100, size=(500, 50))
        y = np.random.randint(100, size=(400, 50))
        df1 = module.DataFrame(x, columns=np.random.shuffle(list(range(x.shape[1]))))
        df2 = module.DataFrame(y, columns=np.random.shuffle(list(range(y.shape[1]))))
        return (df1, df2)

    def test_align_fn_1(dfs):
        df1 = dfs[0]
        df2 = dfs[1]
        df1, df2 = df1.align(df2, axis=1)
        return (df1, df2)

    def test_align_fn_2(dfs):
        df1 = dfs[0]
        df2 = dfs[1]
        df1, df2 = df1.align(df2, axis=1, join='inner')
        return (df1, df2)

    def test_align_fn_3(dfs):
        df1 = dfs[0]
        df2 = dfs[1]
        df1, df2 = df1.align(df2,
                             axis=0,
                             join='left',
                             copy=False,
                             fill_value='test',
                             method='bfill',
                             limit=1,
                             broadcast_axis=0)
        return (df1, df2)

    default_compare_fn = TESTUTIL.default_compare_fn
    TESTUTIL.default_compare_fn = TESTUTIL.run_compare_multiple_df
    TESTUTIL.compare(test_align_fn_1, create_fn=create_align_test_1)
    TESTUTIL.compare(test_align_fn_1, create_fn=create_align_test_2)
    TESTUTIL.compare(test_align_fn_2, create_fn=create_align_test_3)
    TESTUTIL.compare(test_align_fn_3, create_fn=create_align_test_3)
    TESTUTIL.default_compare_fn = default_compare_fn
