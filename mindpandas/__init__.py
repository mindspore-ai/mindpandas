# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
# ============================================================================
"""
Pandas-like interfaces in mindpandas.

Examples:
    >>> import mindpandas as pd

Note:
    - dataframe.py defines all the dataframe interfaces.
"""
import os
import sys

from .config import *
from .dataframe import (DataFrame)
from .general import (to_datetime, concat, date_range, pivot_table)
from .groupby import (DataFrameGroupBy, SeriesGroupBy)
from .io import (read_csv, from_numpy)
from .series import (Series)
from .util import NaT

package_root = os.path.dirname(os.path.abspath(__file__))
if package_root not in sys.path:
    sys.path.insert(1, os.path.join(package_root, 'dist_executor/modules/runtime/python'))

__all__ = ["DataFrame",
           "concat",
           "read_csv",
           "from_numpy",
           "Series",
           "to_datetime",
           "date_range",
           "NaT"]

__all__.sort()
