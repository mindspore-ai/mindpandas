mindpandas.get_min_block_size
======================

.. py:function:: mindpandas.get_min_block_size(**kwargs)

    获取每个分区的当前最小块大小。

    .. note::
        - 获取每个分区的当前最小块大小。

    返回：
        - int，沿每个轴的预期分区数。它是由两个正整数组成的元组。第一个元素是分区的行数，第二个元素是分区的列数。