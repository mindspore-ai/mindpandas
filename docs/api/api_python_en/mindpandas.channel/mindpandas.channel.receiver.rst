.. py:class:: mindpandas.channel.DataReceiver(address, namespace='default', shard_id=0, dataset_name='dataset')

    The receiver (output side) of the channel. It can be used for receiving new object from the channel.

    Parameters：
        - **address** (str) - The ip address of the node current receiver runs on.
        - **namespace** (str, optional) - he namespace that the channel belongs to. By default the value is `default` and the receiver will be running in namespace `default`. DataSender and DataReceiver in different namespaces cannot connect to each other.
        - **shard_id** (int, optional) - Specifies the shard of data that is received by current receiver. By default the value is 0 and the receiver will get data from the shard with id 0.
        - **dataset_name** (str, optional) - The name of the dataset. By default the value is `dataset`.

    .. note::
        Distributed executor has to be started and a DataSender has to be initialized in advance. To pair with the correct DataSender, the `namespace` and `dataset_name` have to be identical to the DataSender.

    .. py:method:: recv()

        Get data from the channel.

        Returns：
            object，the least recent object in the shard that haven't been consumed.

        Raises：
            - **ValueError** - When the `shard_id` of current receiver is invalid.

    .. py:method:: shard_id
        :property:

        Returns the `shard_id` of current receiver.

    .. py:method:: num_shards
        :property:

        Returns the `num_shards` of current channel.
