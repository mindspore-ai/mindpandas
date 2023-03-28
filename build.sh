#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e


init_vars() {
    if command -v python3 > /dev/null; then
        PYTHON=python3
    elif command -v python > /dev/null; then
        PYTHON=python
    else
        command python3
    fi

    PROJECT_BASEDIR=$(realpath "$(dirname "$0")")
    VERSION="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
    PYTHON_VERSION_NUM=$(echo "$VERSION" | awk -F'.' '{print $1$2}')
    if [ "${PYTHON_VERSION_NUM}" == "39" ]; then
        DIST_EXECUTOR_MD5="c349b0bb78cd162412aaaa121f828e4a"
        DOWNLOAD_PATH="https://mindpandas.obs.cn-north-4.myhuaweicloud.com/0.2.0/py39/dist_executor.tar.gz"
    elif [ "${PYTHON_VERSION_NUM}" == "38" ]; then
        DIST_EXECUTOR_MD5="aa34d8dae4e2b84ce7a33285fd299da8"
        DOWNLOAD_PATH="https://mindpandas.obs.cn-north-4.myhuaweicloud.com/0.2.0/py38/dist_executor.tar.gz"
    fi
}

rename_wheel() {
    cd "$PROJECT_BASEDIR/output" || exit
    PACKAGE_LIST=$(ls mindpandas-*-any.whl) || exit
    for PACKAGE_ORIG in $PACKAGE_LIST; do
        MINDPANDAS_VERSION=$(echo "$PACKAGE_ORIG" | awk -F'-' '{print $2}')
        PYTHON_VERSION_TAG="cp$PYTHON_VERSION_NUM"
        PYTHON_ABI_TAG="cp$(python3-config --extension-suffix | awk -F'-' '{print $2}')"
        MACHINE_TAG="$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m)"
        PACKAGE_NEW="mindpandas-$MINDPANDAS_VERSION-$PYTHON_VERSION_TAG-$PYTHON_ABI_TAG-$MACHINE_TAG.whl"
        mv "$PACKAGE_ORIG" "$PACKAGE_NEW"
    done
}

write_checksum() {
    cd "$PROJECT_BASEDIR/output" || exit
    PACKAGE_LIST=$(ls mindpandas-*.whl) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

solve_dependency() {
    echo "Solving Dependency"
    if [[ "${BUILD_LOCAL}" != "True" ]]; then
        # Download for user
        solved=0
        if [ -e "dist_executor.tar.gz" ]; then
            executor_md5=$(md5sum dist_executor.tar.gz |awk -F ' ' '{print $1}')
            if [ "${executor_md5}" == "${DIST_EXECUTOR_MD5}" ]; then
                echo "dist_executor md5: ${executor_md5}"
                solved=1
            else
                echo "dist_executor md5 mismatch"
                rm dist_executor.tar.gz
            fi
        fi

        if [ ${solved} -ne 1 ]; then
            echo "Downloading dependency"
            wget ${DOWNLOAD_PATH}
        fi
    fi

    if [ -d mindpandas/dist_executor ];then
        rm -rf mindpandas/dist_executor
    fi

    tar -xzf dist_executor.tar.gz -C mindpandas/
    chmod +wx -R mindpandas/dist_executor
}

clean_files() {
    cd "$PROJECT_BASEDIR" || exit
    rm -rf build/
    rm -rf mindpandas.egg-info
    rm -rf dist_executor.tar.gz
    rm -rf mindpandas/dist_executor
}

build_wheel() {

    cd "$PROJECT_BASEDIR" || exit

    if [ $# -gt 0 ]; then
        if [ "$1" = "clean" ]; then
            echo "start cleaning mindpandas"
            clean_files
            echo "clean mindpandas done"
            exit
        elif [[ "${BUILD_LOCAL}" = "True" ]]; then
            echo "use existing dist_executor.tar.gz"
        else
            echo "unknown command: $1"
            exit
        fi
    fi

    echo "start building mindpandas"

    if ! "$PYTHON" -c 'import sys; assert sys.version_info.major == 3 and sys.version_info.minor in {8, 9}' > /dev/null; then
        echo "Python 3.8 or 3.9 is required. You are running $("$PYTHON" -V)"
        exit 1
    fi

    rm -rf output
    solve_dependency
    "$PYTHON" setup.py bdist_wheel
    if [ ! -x "dist" ]; then
        echo "Build failed"
        exit 1
    fi

    mv dist output

    rename_wheel
    write_checksum
    clean_files

    echo "Build success, output directory is: $PROJECT_BASEDIR/output"
}

show_usage() {
    echo "Build mindpandas"
    echo ""
    echo "usage: build.sh [-hl] [clean]"
    echo ""
    echo "options:"
    echo "  -h          show this help message and exit"
    echo "  -l          build with local dist_executor"
    echo "  clean       clean build files"
}

check_opts() {
    while getopts 'hl' OPT; do
        case "$OPT" in
        h)
            show_usage
            exit 0
            ;;
        l)
            BUILD_LOCAL="True"
            ;;
        \?)
            show_usage
            exit 1
            ;;
        esac
    done
}

check_opts "$@"
init_vars
build_wheel "$@"
