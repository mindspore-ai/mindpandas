# Welcome to MindPandas

[查看中文](./README_CN.md)

<!-- TOC -->

- [What Is MindPandas](#what-is-mindpandas)
    - [Overview](#overview)
    - [Architecture](#architecture)
- [Installation Methods](#installation-methods)
    - [Confirming System Environment Information](#confirming-system-environment-information)
    - [Installing from pip command](#installing-from-pip-command)
    - [Installing from source code](#installing-from-source-code)
- [Installation Verification](#installation-verification)
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

For detailed architecture design, please refer to [official website document](https://www.mindspore.cn/mindpandas/docs/en/master/index.html).

## Installation Methods

### Confirming System Environment Information

The following table lists the environment required for installing, compiling and running MindPandas:

| software |  version   |
| :------: | :-----: |
|  Linux-x86_64 |  Ubuntu \>=18.04<br/>Euler \>=2.9 |
|  Python  | 3.8 |
|  glibc  |  \>=2.25   |

- Make sure libxml2-utils is installed in your environment.
- Please refer to [requirements](https://gitee.com/mindspore/mindpandas/blob/master/requirements.txt) for other third party dependencies.

### Installing from pip command

If you use the pip, please download the whl package from [MindPandas](https://www.mindspore.cn/versions/en) page and install it.

> Installing whl package will download MindPandas dependencies automatically (detail of dependencies is shown in requirements.txt) in the networked state, and other dependencies should be installed manually.

### Installing from source code

Download [source code](https://gitee.com/mindspore/mindpandas), then enter the `mindpandas` directory to run build.sh script.

```shell
git clone https://gitee.com/mindspore/mindpandas.git
cd mindpandas
bash build.sh
```

The package is in output directory after compiled and you can install with pip.

```shell
pip install output/mindpandas-0.1.0-cp38-cp38-linux_x86_64.whl
```

## Installation Verification

Execute the following command in shell. If no `No module named 'mindpandas'` error is reported, the installation is successful.

```shell
python -c "import mindpandas"
```

## Quickstart

First import MindPandas with the following command.

```python
import mindpandas as pd
```

Set the running mode of MindPandas with the following command, which can speed up your MindPandas workflow.

```python
pd.set_concurrency_mode('multithread')
```

The complete example is as follows:

```python
>>> import mindpandas as pd
>>> pd.set_concurrency_mode('multithread')
>>> pd.set_partition_shape((16, 2))
>>> pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
>>> sum = pd_df.sum()
>>> print(sum)
0 5
1 7
2 9
Name: sum, dtype: int64
```

## Docs

More details about installation guide, tutorials and APIs, please see the
[User Documentation](https://www.mindspore.cn/mindpandas/docs/en/master/mindpandas_install.html).

## Contributing

Welcome contributions. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for
more details.

## Release Notes

The release notes, see our [RELEASE](https://gitee.com/mindspore/mindpandas/blob/master/RELEASE.md).

## License

[Apache License 2.0](https://gitee.com/mindspore/mindpandas/blob/master/LICENSE)
