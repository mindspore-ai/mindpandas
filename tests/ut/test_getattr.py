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

from util import TESTUTIL


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_getattr():
    """
    Test getattr
    Description: tests df getattr
    Expectation: same output as pandas df getattr
    """

    def test_get_id(df):
        output = df.id
        return output

    def test_get_docid(df):
        output = df.docid
        return output

    def test_get_column_not_exist(df):
        try:
            df.x
        except AttributeError:
            return pd.DataFrame({'test': ['passed']})
        else:
            return pd.DataFrame({'test': ['failed']})

    def create_df(module):
        df = module.DataFrame({'id': [1, 2, 3, 4], 'docid': [101, 102, 110, 123]})
        return df

    TESTUTIL.compare(test_get_id, create_df)
    TESTUTIL.compare(test_get_docid, create_df)
    TESTUTIL.compare(test_get_column_not_exist, create_df)
