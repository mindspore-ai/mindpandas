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

import pandas as pd
import mindpandas as mpd


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_iterrows():
    """
    Test iterrows
    Description: tests df.iterrows
    Expectation: same output as pandas df.iterrows
    """

    def create_df():
        pdf = pd.DataFrame([[2, 4, 6, 8],
                            [3, 6, 9, 12],
                            [4, 8, 12, 16],
                            [5, 10, 15, 20],
                            [6, 12, 18, 24]], columns=['A', 'B', 'C', 'D'])
        mdf = mpd.DataFrame(pdf)
        return pdf, mdf

    def test_iterrows_1():
        pdf, mdf = create_df()
        p_iterator = pdf.iterrows()
        m_iterator = mdf.iterrows()
        assert next(p_iterator)[1].equals(next(m_iterator)[1].to_pandas())

    test_iterrows_1()
