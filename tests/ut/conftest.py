import pytest

import mindpandas as mpd


@pytest.fixture(params=["multithread"])
def set_mode(request):
    concurrency_mode = request.param
    if mpd.get_concurrency_mode() != concurrency_mode:
        mpd.set_concurrency_mode(concurrency_mode)
    print(f"\ncurrent concurrency mode:{concurrency_mode}")


@pytest.fixture(params=[(2, 2), (2, 1), (1, 2)])
def set_shape(request):
    partition_shape = request.param
    if mpd.get_partition_shape() != partition_shape:
        mpd.set_partition_shape(partition_shape)
    print(f"\ncurrent partition shape:{partition_shape}")
