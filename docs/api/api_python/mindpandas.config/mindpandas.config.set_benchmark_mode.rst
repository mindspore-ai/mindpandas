.. py:function:: mindpandas.config.set_benchmark_mode(mode)

    用户可以选择是否打开"benchmark"模式进行性能分析。

    参数：
        - **mode** (str) - 可设置为True或False。为True时开启"benchmark"模式，为False时关闭"benchmark"模式。默认为False。

    异常：
        - **ValueError** - `mode` 不是True或False。