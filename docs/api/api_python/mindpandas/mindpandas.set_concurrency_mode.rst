.. py:function:: mindpandas.set_concurrency_mode(mode, **kwargs)

    设置并发模式，可选并发模式有"multithread"和"multiprocess"，默认模式为"multithread"。两种模式的介绍及使用请参考 `MindPandas执行模式介绍及配置说明 <https://www.mindspore.cn/mindpandas/docs/zh-CN/master/mindpandas_configuration.html>`_ 。

    参数：
        - **mode** (str) - 可设置为"multithread"或"multiprocess"。
        - **kwargs** - 在"multithread"模式下运行时不需要额外的参数。在"multiprocess"模式下， `kwargs` 包括：

          - address: 主节点的ip地址，必填。

    异常：
        - **ValueError** - `mode` 不是"multithread"或"multiprocess"。