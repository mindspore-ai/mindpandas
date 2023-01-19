# Copyright 2023 Huawei Technologies Co., Ltd
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
def test_series_items():
    """
    Test Series_item
    Description: tests Series.items()
    Expectation: same output as pandas.Series.items()
    """
    def create_series(n=100):
        pseries = pd.Series(list(range(n)), index=[f'item{i}' for i in range(n)], name='series')
        mseries = mpd.Series(pseries)
        return pseries, mseries

    def test_items():
        pseries, mseries = create_series()
        p_iterator = pseries.items()
        m_iterator = mseries.items()
        assert next(p_iterator) == next(m_iterator)

    test_items()
