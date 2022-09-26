# WelCome to MindPandas


[查看中文](./README_CN.md)

<!-- TOC -->

- [What Is MindPandas](#what-is-mindpandas)
    - [Overview](#overview)
    - [Architecture](#architecture)
- [Installation Methods](#installation-methods)
    - [Pip mode method installation](#pip-mode-method-installation)
    - [Installation by Source Code](#installation-by-source-code)
- [Quickstart](#quickstart)
- [Docs](#docs)
- [Contributing](#contributing)
- [Release Notes](#release-notes)
- [License](#license)

<!-- /TOC -->

## What Is MindPandas

### Overview
MindPandas uses distributed computing engine to accelerate pandas operations, seamlessly integrated and compatible with existing pandas code. Using MindPandas for calculations can use all CPU cores on the computer, which makes MindPandas works especially well on larger datasets.

### Architecture
MindPandas is implemented based on distribution, while native pandas is implemented based on single thread. This means that only one CPU core can be used at a time.

However, MindPandas can use more threads and cores on the machine, or all cores of the entire cluster.

## Installation Methods

### Pip mode method installation

Install pandas via PyPI：

```
pip install mindpandas
```

### Installation by Source Code

1.Download the source code from Gitee

```
git clone https://gitee.com/mindspore/mindpandas.git
```
2.Compile and install mindpandas

Build the whl package for installation, Enter the root directory of the source code, and execute the MindPandas compilation script in the build directory, then execute the command to install the whl package generated in the output directory.

```
cd mindpandas
bash build/build.sh
pip install output/mindpandas-0.1.0-cp37-cp37m-linux_x86_64.whl
```
### Installation Verification

Execute the following command on the Python interactive command line successfully, that is, the installation is successful.

```
import mindpandas as mpd
```

## Quickstart

First install the dependent libraries and packages: pandas, numpy.
Then import MindPandas with the following command.

```python
# import pandas as pd
import mindpandas as pd
```
Set the running mode of MindPandas with the following command, which can speed up your pandas workflow.

```
mindpandas.config.set_concurrency_mode('multithread') 
```
The complete example is as follows:

```python
>>> import mindpandas as mpd
>>> mpd.config.set_concurrency_mode('multithread')
>>> mpd.set_partition_shape((16, 2))
>>> mpd_df = mpd.DataFrame([[1, 2, 3], [4, 5, 6]])
>>> sum = mpd_df.sum()
>>> print(sum)
0 5
1 7
2 9
Name: sum, dtype: int64
```
## Docs

More details about installation guide, tutorials and APIs, please see the
[User Documentation]().

## Contributing

Welcome contributions. See our [Contributor Wiki](https://gitee.com/mindspore/mindpandas/blob/master/CONTRIBUTING.md) for
more details.

## Release Notes

The release notes, see our [RELEASE](https://gitee.com/mindspore/mindpandas/blob/master/RELEASE.md).

## License

[Apache License 2.0](https://gitee.com/mindspore/mindpandas/blob/master/LICENSE)
