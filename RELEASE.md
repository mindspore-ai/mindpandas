# MindPandas Release Notes

## MindPandas 0.2.0 Release Notes

### Major Features and Enhancements

- [STABLE] Provided a shared-memory-based data pipeline, enabling data to be transmitted from the MindPandas data processing process to the MindSpore training process without the need for disk storage, effectively resolving the issue of data analysis and AI training framework separation.
- [STABLE] Streamlined the usage of single-machine multi-process mode, eliminating the need for manual deployment of distributed computing engines.
- [STABLE] Added support for Python 3.9.
- [STABLE] Introduced disk storage functionality for instances where datamem usage exceeds a pre-determined threshold.
- [STABLE] Optimized the cold start time of the distributed computing engine.
- [STABLE] Enhanced memory usage optimization in multi-process mode.
- [STABLE] Refactored the statistical and comparison class APIs, resulting in improved performance.

### API Change

- [STABLE] Added API `mindpandas.DataFrame.memory_usage`.
- [STABLE] Added API `mindpandas.DataFrame.count`.
- [STABLE] Added API `mindpandas.DataFrame.product`.
- [STABLE] Added API `mindpandas.DataFrame.var`.
- [STABLE] Added API `mindpandas.DataFrame.prod`.
- [STABLE] Added API `mindpandas.Series.prod`.
- [STABLE] Added API `mindpandas.Series.isin`.
- [STABLE] Added API `mindpandas.Series.item`.
- [STABLE] Added API `mindpandas.Series.cummin`.
- [STABLE] Added API `mindpandas.Series.count`.
- [STABLE] Added API `mindpandas.Series.cummax`.

### Bug Fixes

- [BUGFIX] Fixed the issue where `read_csv` could not process URLs.
- [BUGFIX] Fixed the issue where the `drop` API produced incorrect calculation results in certain cases.
- [BUGFIX] Fixed the issue where an error occurs when the distributed computing engine process is started using `yrctl start` and you need to manually exit.
- [BUGFIX] Fixed the issue where the distributed computing engine could not be started after setting the proxy.

### Contributors

Thanks goes to these wonderful people:

caiyimeng, chenyue li, dessyang, liyuxia, lichen_101010, Martin Yang, panfengfeng, RobinGrosman, shenghong96, Tom Chen, wangyue, weisun092, xiaohanzhang, xutianyu, yanghaitao, youtianming

Contributions of any kind are welcome!

## MindPandas 0.1.0 Release Notes

MindPandas is a data analysis framework that is compatible with the Pandas interface and provides a data analysis framework with distributed processing capabilities. MindPandas is dedicated to providing high performance tabular data processing capabilities for large volumes of data. MindPandas can be seamlessly integrated into the training process, enabling MindSpore to support the entire training process of a complete AI model.

### Main Features

#### MindPandas

- [STABLE] MindPandas provides over 100 distributed pandas APIs. Modify a small amount of code to switch from native Pandas to MindPandas.
- [STABLE] Provides multi-process and multi-thread execution modes, and provides parallel processing capability of data in single-node or cluster mode to improve data processing performance.
- [STABLE] Efficiently use cluster resources to process large-scale data, resolving the problem that Pandas cannot process large amount of data due to memory limitations.

### Contributors

Thanks goes to these wonderful people:

caiyimeng, chenyue li, dessyang, liyuxia, lichen_101010, Martin Yang, panfengfeng, RobinGrosman, shenghong96, Tom Chen, wangyue, weisun092, xiaohanzhang, xutianyu, yanghaitao, youtianming

Contributions of any kind are welcome!
