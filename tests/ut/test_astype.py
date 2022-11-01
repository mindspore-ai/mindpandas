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
from pandas import CategoricalDtype

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_astype():
    """
    Test astype
    Description: tests df.astype
    Expectation: same output as pandas df.astype
    """

    def create_df_astype(module):
        df = module.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        return df

    def create_ser_astype(module):
        ser = module.Series([1, 2], dtype='int32')
        return ser

    def create_ser_none_astype(module):
        ser = module.Series([True, False, None])
        return ser

    def test_cast_to_float(df):
        return df.astype('float32')

    def test_cast_to_dict(df):
        return df.astype({'col1': 'float32'})

    def test_cast_to_category1(df):
        return df.astype('category')

    def test_cast_to_category2(df):
        cat_dtype = CategoricalDtype(categories=[3, 2, 1], ordered=True)
        return df.astype(cat_dtype)

    TESTUTIL.compare(test_cast_to_float, create_fn=create_df_astype)
    TESTUTIL.compare(test_cast_to_dict, create_fn=create_df_astype)
    TESTUTIL.compare(test_cast_to_category1, create_fn=create_df_astype)
    TESTUTIL.compare(test_cast_to_category2, create_fn=create_df_astype)

    TESTUTIL.compare(test_cast_to_float, create_fn=create_ser_astype)
    TESTUTIL.compare(test_cast_to_category1, create_fn=create_ser_astype)
    TESTUTIL.compare(test_cast_to_category2, create_fn=create_ser_astype)
    TESTUTIL.compare(test_cast_to_float, create_fn=create_ser_none_astype)
    TESTUTIL.compare(test_cast_to_category1, create_fn=create_ser_none_astype)
    TESTUTIL.compare(test_cast_to_category2, create_fn=create_ser_none_astype)
