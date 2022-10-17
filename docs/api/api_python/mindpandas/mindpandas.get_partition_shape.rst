.. py:function:: mindpandas.get_partition_shape()

    获取当前切片维度。

    .. note::
        - 可以设置mode为multithread或yr两种后端模式，默认值为multithread。multithread模式为多线程后端，yr模式为多进程后端。

    返回：
        - Tuple，沿每个轴的预期分区数。它是由两个正整数组成的元组。第一个元素是分区的行数，第二个元素是分区的列数。