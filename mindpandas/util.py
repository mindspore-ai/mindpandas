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
"""Mindpandas Utility Functions"""
import codecs
import tempfile
from io import (
    BufferedIOBase,
    RawIOBase,
)

import pandas


def hashable(obj):
    # check if object is hashable
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def is_file_like(obj):
    # check if object is file_like
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False
    return bool(hasattr(obj, "__iter__"))


def is_binary_mode(ms_handle, mode):
    """Used to check if the handle is opened in binary mode"""
    if "t" in mode or "b" in mode:
        return "b" in mode

    # some exceptions
    ms_text_classes = (codecs.StreamWriter, codecs.StreamReader,
                       codecs.StreamReaderWriter, tempfile.SpooledTemporaryFile)
    if issubclass(type(ms_handle), ms_text_classes):
        return False

    ms_binary_classes = (BufferedIOBase, RawIOBase)
    return isinstance(ms_handle, ms_binary_classes) or "b" in getattr(ms_handle, "mode", mode)


def is_boolean(obj):
    # check if object is instance of 'bool'
    return isinstance(obj, bool)


def is_list(obj):
    # check if object is list
    return isinstance(obj, list)


def is_boolean_array(obj):
    """
    Check if object is a boolean array.
    """
    return is_list(obj) and all(map(is_boolean, obj))


def is_range_like(obj):
    """
    Used to check if the object is range-like.
    """
    return (
        hasattr(obj, "__iter__")
        and hasattr(obj, "start")
        and hasattr(obj, "stop")
        and hasattr(obj, "step")
    )


def is_full_grab_slice(slc, sequence_len=None):
    """
    Used to check weather the passed slice grabs the whole sequence or not.
    """
    assert isinstance(slc, slice), "slice object required"
    return (
        slc.start in (None, 0)
        and slc.step in (None, 1)
        and (
            slc.stop is None or (sequence_len is not None and slc.stop >= sequence_len)
            )
    )


NaT = pandas.NaT

NO_VALUE = NotImplemented
