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
import mindpandas as mpd


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_where():
    """
    Test where
    Description: tests df.where
    Expectation: same output as pandas df.where
    """

    def create_df_other_is_scalar():
        df = pd.DataFrame(np.arange(100).reshape(10, 10), columns=list('ABCDEFGHIJ'))
        cond = df % 3 == 0
        other = np.nan
        return df, cond, other

    def create_df_other_is_df():
        df = pd.DataFrame(np.arange(100).reshape(10, 10), columns=list('ABCDEFGHIJ'))
        cond = df % 3 == 0
        other = pd.DataFrame(np.full((10, 10), -1), columns=list('ABCDEFGHIJ'))
        return df, cond, other

    def test_1():
        df, cond, other = create_df_other_is_scalar()
        mdf = mpd.DataFrame(df)
        mcond = mpd.DataFrame(cond)
        ms_where = mdf.where(mcond, other)
        pandas_where = df.where(cond, other)
        assert pandas_where.equals(ms_where.to_pandas())

    def test_2():
        df, cond, other = create_df_other_is_df()
        mdf = mpd.DataFrame(df)
        mcond = mpd.DataFrame(cond)
        mo = mpd.DataFrame(other)
        ms_where = mdf.where(mcond, mo)
        pandas_where = df.where(cond, other)
        assert pandas_where.equals(ms_where.to_pandas())

    test_1()
    test_2()
