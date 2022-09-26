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
import pandas as pd
import pytest

import mindpandas as mpd


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_dtype():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    df = pd.DataFrame(arr, dtype=np.float32)
    ms_df = mpd.DataFrame(arr, dtype=np.float32)
    assert ms_df.dtypes[0], df.dtypes[0]


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_copy():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    ms_df = mpd.DataFrame(arr)
    ms_df_copy = mpd.DataFrame(arr, copy=True)
    df = mpd.DataFrame(arr)
    df_copy = mpd.DataFrame(arr, copy=True)

    arr[0][0] = 10

    assert ms_df[0][0] == df[0][0]
    assert ms_df_copy[0][0] == df_copy[0][0]


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_series_copy():
    d = {'a': 1, 'b': 2, 'c': 3}
    ser = pd.Series(d, index=['a', 'b', 'c'])
    ser_copy = pd.Series(d, index=['a', 'b', 'c'], copy=True)
    ms_ser = mpd.Series(d, index=['a', 'b', 'c'])
    ms_ser_copy = mpd.Series(d, index=['a', 'b', 'c'], copy=True)

    d[0] = 10

    assert ser[0] == ms_ser[0]
    assert ser_copy[0] == ms_ser_copy[0]
