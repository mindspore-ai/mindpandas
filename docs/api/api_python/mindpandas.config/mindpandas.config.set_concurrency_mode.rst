.. py:function:: mindpandas.config.set_concurrency_mode(mode, **kwargs)

    设置并发模式，可选并发模式有"multithread"和"multiprocess"，默认模式为"multithread"。两种模式的介绍及使用请参考 `MindPandas执行模式介绍及配置说明 <https://www.mindspore.cn/mindpandas/docs/zh-CN/master/mindpandas_configuration.html>`_ 。

    参数：
        - **mode** (str) - 可设置为"multithread"或"multiprocess"。
        - **\*\*kwargs** - 在"multithread"模式下运行时不需要额外的参数。在"multiprocess"模式下， `kwargs` 包括：

          - address (str) : master节点的IP地址。可选，默认使用"127.0.0.1"。
          - cpu (int) : 用户设定使用的CPU核数。可选，默认使用当前节点的所有核。
          - datamem (int) : 共享内存的大小，单位是MB。可选，默认使用当前空闲内存的30%。
          - mem (int) : MindPandas使用的总内存（包含共享内存），单位是MB。可选，默认使用当前空闲内存的90%。
          - tmp_dir (str) : 临时文件的存放路径。可选，默认使用"/tmp/mindpandas/"作为临时文件的存放路径。
          - tmp_file_size_limit (int) : 临时文件的大小限制，单位是MB。可选，默认上限为当前空闲磁盘空间的95%。

    异常：
        - **ValueError** - `mode` 不是"multithread"或"multiprocess"。