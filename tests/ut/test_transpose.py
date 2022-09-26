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
def test_transpose():
    """
    Test transpose
    Description: tests df.transpose
    Expectation: same output as pandas df.transpose
    """

    def create_df_square_large(module):
        df = module.DataFrame(np.arange(0, 10000).reshape((100, 100)),
                              columns=[f"col{i}" for i in range(1, 101)])
        return df

    def create_df_nonsquare_large(module):
        df = module.DataFrame(np.arange(0, 10000).reshape((500, 20)),
                              columns=[f"col{i}" for i in range(1, 21)])
        return df

    def test_transpose_fn(df):
        return df.T

    TESTUTIL.compare(test_transpose_fn, create_fn=TESTUTIL.create_df_small)
    TESTUTIL.compare(test_transpose_fn, create_fn=create_df_square_large)
    TESTUTIL.compare(test_transpose_fn, create_fn=create_df_nonsquare_large)
