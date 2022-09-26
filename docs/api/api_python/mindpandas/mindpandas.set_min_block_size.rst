mindpandas.set_min_block_size
======================

.. py:function:: mindpandas.set_min_block_size(min_block_size)


    .. note::
        - 用户可以使用此API设置每个分区的最小块大小。它表示每个轴的最小尺寸。每个分区的大小将大于或等于（min_block_size，min_block _size），除非原始数据小于此大小。例如，如果min_block_size设置为32，只有16列的数据，分区形状为（2,2），然后在分区期间我们不会进一步拆分列。

    参数：
        - **min_block_size** (int) - 分区的最小块的最小行数和列数。

    异常：
        - **ValueError** - min_block_size不是int类型。