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
def test_isna():
    """
    Test isna
    Description: tests df.isna
    Expectation: same output as pandas df.isna
    """

    def test_isna_fn(df):
        df = df.isna()
        return df

    def create_series_range_nan(module):
        ser = TESTUTIL.create_series_range(module)
        ls1 = [0, 200, 400, 600, 800]
        ls2 = [100, 300, 500, 700, 900]
        for i in ls1:
            ser[i] = None
        for i in ls2:
            ser[i] = np.nan
        return ser

    def create_series_unique_nan(module):
        ser = module.Series([None])
        return ser

    TESTUTIL.compare(test_isna_fn, TESTUTIL.create_df_range_float)
    # test Series.isna
    TESTUTIL.compare(test_isna_fn, TESTUTIL.create_series_range)
    TESTUTIL.compare(test_isna_fn, TESTUTIL.create_series_unique)
    TESTUTIL.compare(test_isna_fn, TESTUTIL.create_series_dup)
    TESTUTIL.compare(test_isna_fn, create_series_range_nan)
    TESTUTIL.compare(test_isna_fn, create_series_unique_nan)
