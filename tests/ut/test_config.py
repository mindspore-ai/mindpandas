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
import pytest

import mindpandas as mpd


@pytest.fixture(autouse=True)
def restore_default_config():
    mpd.config.set_concurrency_mode("multithread")
    mpd.config.set_adaptive_concurrency(False)
    mpd.config.set_partition_shape((16, 16))
    mpd.config.set_min_block_size(1)


def test_concurrency_mode():
    assert mpd.config.get_concurrency_mode() == 'multithread'


def test_partition_shape():
    assert mpd.config.get_partition_shape() == (16, 16)

    mpd.config.set_partition_shape((8, 16))
    assert mpd.config.get_partition_shape() == (8, 16)


def test_min_block_size():
    assert mpd.config.get_min_block_size() == 1

    mpd.config.set_min_block_size(8)
    assert mpd.config.get_min_block_size() == 8


def test_default_config():
    """
    Feature: default config
    Description: Revert to default configuration after last test case
    Expectation: success
    """
    print("reset to default config")


if __name__ == '__main__':
    test_min_block_size()
    test_partition_shape()
    test_min_block_size()
