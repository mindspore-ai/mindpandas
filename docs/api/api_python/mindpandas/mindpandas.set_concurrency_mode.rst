mindpandas.set_concurrency_mode
======================

.. py:function:: mindpandas.set_concurrency_mode(mode, **kwargs)

    设置后端运行模式。

    .. note::
        - 设置后端运行模式，默认为“multithread”模式，当然还有“yr”模式可供选择。在“yr”模式下运行时，必须先部署集群，详细操作方法请参考[MindPandas后端执行模式配置及性能介绍](https://www.mindspore.cn/mindpandas/docs/zh-CN/master/mindpandas_configuration.html)。

    参数：
        - **mode** (str) - 可设置为“multithread”或“yr”。
        - **kwargs** - 在“multithread”模式下运行时不需要额外的参数。在“yr”模式下，kwargs包括：
                        *server_address：主节点的ip地址，必填。
                        *ds_address：主节点的ip地址，必填。

    异常：
        - **ValueError** - “mode”不是“multithread”或“yr”。