.. py:class:: mindpandas.channel.DataReceiver(address, namespace='default', shard_id=0, dataset_name='dataset')

    从channel接收数据的类。负责接收来自channel的新对象。

    参数：
        - **address** (str) - 当前receiver运行的节点的ip地址。
        - **namespace** (str, 可选) - channel所属的命名空间。默认值：`default` ，receiver将在命名空间 `default` 中运行。不同命名空间的DataSender和DataReceiver不能相互连接。
        - **shard_id** (int, 可选) - 指定当前receiver接收数据集的哪个分片。默认值：0，receiver将从id为0的分片获取数据。
        - **dataset_name** (str, 可选) - 数据集的名称。默认值：`dataset` 。

    .. note::
        必须提前启动分布式执行引擎，并且提前初始化DataSender。要与正确的DataSender配对，`namespace` 和 `dataset_name` 必须与DataSender相同。

    .. py:method:: recv()

        通过channel获取数据。

        返回：
            object，分片中最近没有被消费的对象。

        异常：
            - **ValueError** - 当前receiver的 `shard_id` 无效时。

    .. py:method:: shard_id
        :property:

        返回当前receiver的 `shard_id` 。

    .. py:method:: num_shards
        :property:

        返回当前channel的 `num_shards` 。
