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
Module for eager execution of general pandas operations.
"""
import numpy as np
import pandas
import mindpandas.internal_config as i_config
from .eager_frame import EagerFrame
from .eager_backend import get_partition

def concat(objs,
           axis=0,
           obj_is_series=None,
           join='outer',
           ignore_index=False,
           verify_integrity=False,
           sort=False,
           keys=None,
           names=None,
           levels=None,
           **kwargs):
    '''Eager concatenation of objects using specified parameters.'''
    pd_df_list = []
    for i, obj in enumerate(objs):
        force_series = False
        if obj_is_series[i]:
            force_series = True
        # If object is series, set force_series=True for to_pandas()
        pd_df = obj.to_pandas(force_series=force_series)
        pd_df_list.append(pd_df)
    pd_concated_df = pandas.concat(pd_df_list,
                                   axis=axis,
                                   join=join,
                                   ignore_index=ignore_index,
                                   verify_integrity=verify_integrity,
                                   sort=sort,
                                   keys=keys,
                                   names=names,
                                   levels=levels,
                                   **kwargs)
    partition = get_partition().put(data=pd_concated_df, coord=(0, 0))
    if isinstance(pd_concated_df, pandas.DataFrame):
        output_frame = EagerFrame(np.array([[partition]]), pd_concated_df.index, pd_concated_df.columns)
    else:
        output_frame = EagerFrame(np.array([[partition]]), pd_concated_df.index)
    output_frame = output_frame.repartition(i_config.get_partition_shape(),
                                            i_config.get_min_block_size())
    return output_frame
