.. py:function:: mindpandas.config.set_benchmark_mode(mode)

    用户可以选择是否打开"benchmark"模式进行性能分析。

    参数：
        - **mode** (bool) - 可设置为 ``True`` 或 ``False``。为 ``True`` 时开启"benchmark"模式，为 ``False`` 时关闭"benchmark"模式。默认为 ``False``。

    异常：
        - **ValueError** - `mode` 不是bool。