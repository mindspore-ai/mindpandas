import pandas as pd
import pytest

import mindpandas as mpd


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_partition_df_1():
    """
    Test DataFrame partition.
    """
    mpd.set_partition_shape((2, 2))
    df = pd.DataFrame({"A": [1, 2, 3, 4],
                       "B": [5, 6, 7, 8],
                       "C": [9, 10, 11, 12],
                       "D": [13, 14, 15, 16]})
    p00 = pd.DataFrame({"A": [1, 2],
                        "B": [5, 6]}, index=[0, 1])
    p01 = pd.DataFrame({"C": [9, 10],
                        "D": [13, 14]}, index=[0, 1])
    p10 = pd.DataFrame({"A": [3, 4],
                        "B": [7, 8]}, index=[2, 3])
    p11 = pd.DataFrame({"C": [11, 12],
                        "D": [15, 16]}, index=[2, 3])

    mdf = mpd.DataFrame(df)

    assert mdf.backend_frame.partition_shape == (2, 2)

    assert mdf.backend_frame.partitions[0, 0].get().equals(p00)
    assert mdf.backend_frame.partitions[0, 1].get().equals(p01)
    assert mdf.backend_frame.partitions[1, 0].get().equals(p10)
    assert mdf.backend_frame.partitions[1, 1].get().equals(p11)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_partition_df_2():
    mpd.set_partition_shape((3, 3))
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8],
                       "B": [1, 2, 3, 4, 5, 6, 7, 8],
                       "C": [1, 2, 3, 4, 5, 6, 7, 8],
                       "D": [1, 2, 3, 4, 5, 6, 7, 8]})

    p00 = pd.DataFrame({"A": [1, 2, 3],
                        "B": [1, 2, 3]}, index=[0, 1, 2])
    p01 = pd.DataFrame({"C": [1, 2, 3]}, index=[0, 1, 2])
    p02 = pd.DataFrame({"D": [1, 2, 3]}, index=[0, 1, 2])
    p10 = pd.DataFrame({"A": [4, 5, 6],
                        "B": [4, 5, 6]}, index=[3, 4, 5])
    p11 = pd.DataFrame({"C": [4, 5, 6]}, index=[3, 4, 5])
    p12 = pd.DataFrame({"D": [4, 5, 6]}, index=[3, 4, 5])
    p20 = pd.DataFrame({"A": [7, 8],
                        "B": [7, 8]}, index=[6, 7])
    p21 = pd.DataFrame({"C": [7, 8]}, index=[6, 7])
    p22 = pd.DataFrame({"D": [7, 8]}, index=[6, 7])

    mdf = mpd.DataFrame(df)

    assert mdf.backend_frame.partition_shape == (3, 3)

    assert mdf.backend_frame.partitions[0, 0].get().equals(p00)
    assert mdf.backend_frame.partitions[0, 1].get().equals(p01)
    assert mdf.backend_frame.partitions[0, 2].get().equals(p02)
    assert mdf.backend_frame.partitions[1, 0].get().equals(p10)
    assert mdf.backend_frame.partitions[1, 1].get().equals(p11)
    assert mdf.backend_frame.partitions[1, 2].get().equals(p12)
    assert mdf.backend_frame.partitions[2, 0].get().equals(p20)
    assert mdf.backend_frame.partitions[2, 1].get().equals(p21)
    assert mdf.backend_frame.partitions[2, 2].get().equals(p22)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_partition_series():
    mpd.set_partition_shape((2, 2))
    ser = pd.Series([1, 2, 3, 4, 5, 6])

    mser = mpd.Series(ser)
    p00 = pd.DataFrame({"__unsqueeze_series__": [1, 2, 3]}, index=[0, 1, 2])
    p10 = pd.DataFrame({"__unsqueeze_series__": [4, 5, 6]}, index=[3, 4, 5])

    assert mser.backend_frame.partition_shape == (2, 1)
    assert mser.backend_frame.partitions[0, 0].get().equals(p00)
    assert mser.backend_frame.partitions[1, 0].get().equals(p10)


@pytest.mark.usefixtures("set_mode", "set_shape")
def test_repartition_df():
    mpd.set_partition_shape((2, 2))
    df = pd.DataFrame({"A": [1, 2, 3, 4],
                       "B": [5, 6, 7, 8],
                       "C": [9, 10, 11, 12],
                       "D": [13, 14, 15, 16]})

    mdf = mpd.DataFrame(df)
    mdf.repartition((3, 3))

    p00 = pd.DataFrame({"A": [1, 2],
                        "B": [5, 6]}, index=[0, 1])
    p01 = pd.DataFrame({"C": [9, 10]}, index=[0, 1])
    p02 = pd.DataFrame({"D": [13, 14]}, index=[0, 1])
    p10 = pd.DataFrame({"A": [3],
                        "B": [7]}, index=[2])
    p11 = pd.DataFrame({"C": [11]}, index=[2])
    p12 = pd.DataFrame({"D": [15]}, index=[2])
    p20 = pd.DataFrame({"A": [4],
                        "B": [8]}, index=[3])
    p21 = pd.DataFrame({"C": [12]}, index=[3])
    p22 = pd.DataFrame({"D": [16]}, index=[3])

    assert mdf.backend_frame.partition_shape == (3, 3)

    assert mdf.backend_frame.partitions[0, 0].get().equals(p00)
    assert mdf.backend_frame.partitions[0, 1].get().equals(p01)
    assert mdf.backend_frame.partitions[0, 2].get().equals(p02)
    assert mdf.backend_frame.partitions[1, 0].get().equals(p10)
    assert mdf.backend_frame.partitions[1, 1].get().equals(p11)
    assert mdf.backend_frame.partitions[1, 2].get().equals(p12)
    assert mdf.backend_frame.partitions[2, 0].get().equals(p20)
    assert mdf.backend_frame.partitions[2, 1].get().equals(p21)
    assert mdf.backend_frame.partitions[2, 2].get().equals(p22)


if __name__ == '__main__':
    test_partition_df_1()
    test_partition_df_2()
    test_partition_series()
